"""
VoiceBank - Flask server
Voice-first flows for create account, login, and a spoken account menu.
"""
import threading
from functools import wraps

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

import bank_core

app = Flask(__name__)
CORS(app)

bank_core.init_db()

flow_lock = threading.Lock()


def exclusive(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not flow_lock.acquire(blocking=False):
            return jsonify({"success": False, "message": "The assistant is busy. Please wait a moment."})
        try:
            return fn(*args, **kwargs)
        finally:
            flow_lock.release()
    return wrapper


def fmt_amount(value):
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


def speak_async(text):
    """For the couple of fixed strings the FRONT-END sends verbatim to
    /api/speak (index.html)."""
    threading.Thread(target=bank_core.speak, args=(text,), daemon=True).start()


def say_async(key, **kwargs):
    """For backend-originated announcements that don't need to block the
    HTTP response (e.g. confirming a language switch made by tapping a
    button rather than by voice)."""
    threading.Thread(target=bank_core.say, args=(key,), kwargs=kwargs, daemon=True).start()


def get_amount_by_voice(ask_key):
    """Asks for and confirms a rupee amount. Money is the ONE thing this
    app always confirms - the yes/no check already understands 'correct',
    'crt', 'sahi', 'sari', 'avunu', 'howdu', 'yes', and more, in whichever
    language is active."""
    for _ in range(3):
        bank_core.say(ask_key)
        heard = bank_core.listen(max_wait=12, silence_end=1.8)
        amount = bank_core.parse_amount(heard)
        if amount and amount > 0:
            if bank_core.ask_yes_no("amount_confirm", amount=fmt_amount(amount)):
                return amount
        elif heard:
            bank_core.say("amount_not_understood")
    return None


def run_deposit(account_id, typed_amount=""):
    if typed_amount:
        try:
            amount = float(typed_amount)
        except ValueError:
            bank_core.say("amount_not_understood")
            return {"success": False, "message": "Invalid amount"}
    else:
        amount = get_amount_by_voice("ask_deposit_amount")

    if amount is None:
        bank_core.say("deposit_cancelled")
        return {"success": False, "message": "Deposit cancelled."}
    if amount <= 0:
        bank_core.say("invalid_amount")
        return {"success": False, "message": "Amount must be greater than zero."}

    success, new_balance = bank_core.deposit(account_id, amount)
    bank_core.say("deposit_success", amount=fmt_amount(amount), balance=fmt_amount(new_balance))
    return {"success": success,
            "message": f"Deposited {fmt_amount(amount)} rupees. New balance: {fmt_amount(new_balance)} rupees."}


def run_withdraw(account_id, typed_amount=""):
    if typed_amount:
        try:
            amount = float(typed_amount)
        except ValueError:
            bank_core.say("amount_not_understood")
            return {"success": False, "message": "Invalid amount"}
    else:
        amount = get_amount_by_voice("ask_withdraw_amount")

    if amount is None:
        bank_core.say("withdraw_cancelled")
        return {"success": False, "message": "Withdrawal cancelled."}
    if amount <= 0:
        bank_core.say("invalid_amount")
        return {"success": False, "message": "Amount must be greater than zero."}

    # Re-verifies BOTH face and voice before releasing any cash - same
    # camera/beep flow used at login, run again here on purpose.
    if not bank_core.verify_face(account_id):
        bank_core.say("face_verification_failed_withdraw")
        return {"success": False, "message": "Face verification failed. Withdrawal cancelled."}
    if not bank_core.verify_withdrawal_voice(account_id):
        return {"success": False, "message": "Voice verification failed. Withdrawal cancelled."}

    success, new_balance, error = bank_core.withdraw(account_id, amount)
    if not success:
        if error == "insufficient":
            bank_core.say("insufficient_balance", balance=fmt_amount(new_balance))
            return {"success": False, "message": f"Insufficient balance. Current balance: {fmt_amount(new_balance)} rupees."}
        bank_core.say("invalid_amount")
        return {"success": False, "message": "Amount must be greater than zero."}

    bank_core.play_cash_dispensing_sound()
    bank_core.say("withdraw_success", amount=fmt_amount(amount), balance=fmt_amount(new_balance))
    return {"success": True,
            "message": f"Withdrew {fmt_amount(amount)} rupees. Remaining balance: {fmt_amount(new_balance)} rupees."}


def run_balance(account_id):
    balance = bank_core.get_balance(account_id)
    bank_core.say("balance_readout", amount=fmt_amount(balance))
    return balance


def run_statement(account_id):
    txs = bank_core.mini_statement(account_id)
    lang = bank_core.get_language()
    if not txs:
        bank_core.say("no_transactions")
    else:
        bank_core.say("statement_intro", count=len(txs))
        for kind, amount, when in txs:
            kind_label = bank_core.KIND_LABELS.get(kind, {}).get(lang, kind)
            bank_core.say("statement_line", kind=kind_label, amount=fmt_amount(amount), when=when)
    return [{"type": k, "amount": a, "when": w} for k, a, w in txs]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/languages', methods=['GET'])
def api_languages():
    return jsonify({
        "languages": [{"key": k, "label": label} for k, label in bank_core.available_languages()],
        "current": bank_core.get_language(),
    })


@app.route('/api/set_language', methods=['POST'])
def api_set_language():
    data = request.json or {}
    key = (data.get('language') or "").strip().lower()
    if bank_core.set_language(key):
        say_async("language_set")
        return jsonify({"success": True, "language": key})
    return jsonify({"success": False, "message": "Unknown language."})


@app.route('/api/speak', methods=['POST'])
def api_speak():
    data = request.json or {}
    text = data.get('text', '')
    if text:
        speak_async(text)
    return jsonify({"spoken": text})


@app.route('/api/language_flow', methods=['POST'])
@exclusive
def api_language_flow():
    """The very first thing spoken when the page loads: which language is
    the person comfortable with."""
    try:
        key = bank_core.ask_language_choice(attempts=3)
    except Exception as e:
        print(f"[api_language_flow Error]: {e}")
        return jsonify({"success": False, "language": bank_core.get_language()})
    if key:
        bank_core.set_language(key)
        bank_core.say("language_set")
        return jsonify({"success": True, "language": key})
    bank_core.say("language_not_understood")
    return jsonify({"success": False, "language": bank_core.get_language()})


@app.route('/api/mode_flow', methods=['POST'])
@exclusive
def api_mode_flow():
    """Asked right after the language is set: create account, or login."""
    try:
        mode = bank_core.ask_mode_choice(attempts=3)
    except Exception as e:
        print(f"[api_mode_flow Error]: {e}")
        return jsonify({"mode": ""})
    if mode in ('create', 'login'):
        return jsonify({"mode": mode})
    return jsonify({"mode": ""})


@app.route('/api/create_account_flow', methods=['POST'])
@exclusive
def create_account_flow():
    data = request.json or {}
    name = (data.get('name') or "").strip()
    pin = (data.get('pin') or "").strip()

    if not name:
        # Accepts what is heard the FIRST time - no confirmation round-trip
        # for the name. Only re-asks if literally nothing was heard.
        for attempt in range(2):
            bank_core.say("ask_name")
            heard = bank_core.listen(max_wait=12, silence_end=1.5)
            if heard:
                name = heard.title()
                break
        if not name:
            bank_core.say("could_not_get_name")
            return jsonify({"success": False, "message": "Could not get your name."})

    if not (len(pin) == 4 and pin.isdigit()):
        pin = bank_core.listen_digits("ask_pin", expected_len=4, max_wait=12, silence_end=2.0, attempts=2)
        if not pin:
            bank_core.say("could_not_get_pin")
            return jsonify({"success": False, "message": "Could not get a valid pin."})

    bank_core.say("creating_account", name=name)
    # Captures the account holder's face (HD, cropped to just the face) and
    # voice here - these are what withdrawal will re-verify against later.
    account_id = bank_core.create_account(name, pin)
    if account_id is None:
        return jsonify({"success": False, "message": "Account was not created. Face or voice registration failed."})

    bank_core.say("account_created", account_id=account_id)
    return jsonify({"success": True, "name": name, "account_id": account_id})


@app.route('/api/login_flow', methods=['POST'])
@exclusive
def login_flow():
    data = request.json or {}
    account_id_raw = (data.get('account_id') or "").strip()
    pin = (data.get('pin') or "").strip()

    if not account_id_raw:
        account_id_raw = bank_core.listen_digits("ask_account_number", expected_len=None,
                                                   max_wait=20, silence_end=2.0, attempts=2)
        if not account_id_raw:
            bank_core.say("could_not_get_account_number")
            return jsonify({"success": False, "message": "Could not get your account number."})

    try:
        account_id = int(account_id_raw)
    except ValueError:
        bank_core.say("could_not_get_account_number")
        return jsonify({"success": False, "message": "Invalid account number."})

    if not (len(pin) == 4 and pin.isdigit()):
        pin = bank_core.listen_digits("ask_pin", expected_len=4, max_wait=12, silence_end=2.0, attempts=2)
        if not pin:
            bank_core.say("could_not_get_pin")
            return jsonify({"success": False, "message": "Invalid pin."})

    bank_core.say("verifying")
    success = bank_core.login(account_id, pin)

    if success:
        bank_core.say("login_success")
        return jsonify({"success": True, "account_id": account_id})

    bank_core.say("login_failed")
    return jsonify({"success": False, "message": "Login failed. Incorrect PIN or face not recognized."})


@app.route('/api/cancel_listen', methods=['POST'])
def api_cancel_listen():
    """Not @exclusive on purpose - must be able to interrupt an in-progress
    /api/voice_menu listen immediately (e.g. clicking Logout mid-question)."""
    bank_core.request_cancel_listen()
    return jsonify({"cancelled": True})


@app.route('/api/voice_menu', methods=['POST'])
@exclusive
def api_voice_menu():
    data = request.json or {}
    account_id = int(data.get('account_id'))
    first = bool(data.get('first'))

    prompt_key = "menu_first" if first else "menu_again"
    # ask_command retries internally (loosening the matching bar each time)
    # before ever surfacing "I did not understand" - see bank_core.py.
    action, matched = bank_core.ask_command(prompt_key, attempts=2)

    if bank_core.cancel_listen_event.is_set():
        bank_core.cancel_listen_event.clear()
        return jsonify({"action": "cancelled", "heard": "", "message": "", "transactions": []})

    result = {"action": action, "heard": matched, "message": "", "transactions": []}

    if action == 'balance':
        balance = run_balance(account_id)
        result["message"] = f"Balance: {fmt_amount(balance)} rupees"
    elif action == 'deposit':
        result["message"] = run_deposit(account_id)["message"]
    elif action == 'withdraw':
        result["message"] = run_withdraw(account_id)["message"]
    elif action == 'statement':
        result["transactions"] = run_statement(account_id)
        result["message"] = "" if result["transactions"] else "No recent transactions."
    elif action == 'logout':
        bank_core.say("logout_command")
        result["message"] = "Logged out."
    elif action == 'unknown':
        bank_core.say("unknown_command", heard=matched)
        result["message"] = f"Did not understand: {matched}"
    else:
        result["message"] = "Nothing was heard."

    return jsonify(result)


@app.route('/api/balance', methods=['POST'])
@exclusive
def api_balance():
    account_id = int((request.json or {}).get('account_id'))
    return jsonify({"success": True, "balance": run_balance(account_id)})


@app.route('/api/deposit', methods=['POST'])
@exclusive
def api_deposit():
    data = request.json or {}
    return jsonify(run_deposit(int(data.get('account_id')), (data.get('amount') or "").strip()))


@app.route('/api/withdraw', methods=['POST'])
@exclusive
def api_withdraw():
    data = request.json or {}
    return jsonify(run_withdraw(int(data.get('account_id')), (data.get('amount') or "").strip()))


@app.route('/api/statement', methods=['POST'])
@exclusive
def api_statement():
    account_id = int((request.json or {}).get('account_id'))
    return jsonify({"success": True, "transactions": run_statement(account_id)})


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)