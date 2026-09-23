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
    threading.Thread(target=bank_core.speak, args=(text,), daemon=True).start()


def get_amount_by_voice(verb):
    for _ in range(2):
        bank_core.speak(f"How much would you like to {verb}? Please say the amount in rupees.")
        heard = bank_core.listen(max_wait=12, silence_end=1.8)
        amount = bank_core.parse_amount(heard)
        if amount and amount > 0:
            if bank_core.ask_yes_no(f"You said {fmt_amount(amount)} rupees. Say correct to confirm, or wrong to say it again."):
                return amount
        elif heard:
            bank_core.speak("I could not understand the amount.")
    return None


def run_deposit(account_id, typed_amount=""):
    if typed_amount:
        try:
            amount = float(typed_amount)
        except ValueError:
            bank_core.speak("I could not understand the amount.")
            return {"success": False, "message": "Invalid amount"}
    else:
        amount = get_amount_by_voice("deposit")
    if amount is None:
        bank_core.speak("Deposit cancelled.")
        return {"success": False, "message": "Deposit cancelled."}
    success, msg = bank_core.deposit(account_id, amount)
    bank_core.speak(msg)
    return {"success": success, "message": msg}


def run_withdraw(account_id, typed_amount=""):
    if typed_amount:
        try:
            amount = float(typed_amount)
        except ValueError:
            bank_core.speak("I could not understand the amount.")
            return {"success": False, "message": "Invalid amount"}
    else:
        amount = get_amount_by_voice("withdraw")
    if amount is None:
        bank_core.speak("Withdrawal cancelled.")
        return {"success": False, "message": "Withdrawal cancelled."}

    if not bank_core.verify_withdrawal_voice(account_id):
        return {"success": False, "message": "Voice verification failed. Withdrawal cancelled."}

    success, msg = bank_core.withdraw(account_id, amount)
    if success:
        bank_core.play_cash_dispensing_sound()
    bank_core.speak(msg)
    return {"success": success, "message": msg}


def run_balance(account_id):
    balance = bank_core.get_balance(account_id)
    bank_core.speak(f"Your current available balance is {balance:.2f} rupees.")
    return balance


def run_statement(account_id):
    txs = bank_core.mini_statement(account_id)
    if not txs:
        bank_core.speak("You have no recent transactions.")
    else:
        parts = [f"Here are your last {len(txs)} transactions."]
        for kind, amount, when in txs:
            parts.append(f"{kind.lower()} of {amount:.2f} rupees on {when}.")
        bank_core.speak(" ".join(parts))
    return [{"type": k, "amount": a, "when": w} for k, a, w in txs]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/speak', methods=['POST'])
def api_speak():
    data = request.json or {}
    text = data.get('text', '')
    if text:
        speak_async(text)
    return jsonify({"spoken": text})


@app.route('/api/mode_flow', methods=['POST'])
@exclusive
def api_mode_flow():
    for _ in range(3):
        bank_core.speak("Say create account to open a new account, or say login if you already have one.")
        heard = bank_core.listen(max_wait=10, silence_end=1.3)
        t = heard.lower()
        if 'create' in t or 'new account' in t or 'open account' in t or 'sign up' in t:
            return jsonify({"mode": "create"})
        if 'login' in t or 'log in' in t or 'sign in' in t:
            return jsonify({"mode": "login"})
        if heard:
            bank_core.speak("I did not catch that.")
    return jsonify({"mode": ""})


@app.route('/api/create_account_flow', methods=['POST'])
@exclusive
def create_account_flow():
    data = request.json or {}
    name = (data.get('name') or "").strip()
    pin = (data.get('pin') or "").strip()

    if not name:
        for attempt in range(2):
            bank_core.speak("Please say your full name.")
            heard = bank_core.listen(max_wait=12, silence_end=1.5)
            if not heard:
                continue
            name = heard.title()
            if attempt == 1 or bank_core.ask_yes_no(f"I heard {name}. Say correct if that is right, or wrong to say it again."):
                break
            name = ""
        if not name:
            bank_core.speak("I could not get your name. Please try again.")
            return jsonify({"success": False, "message": "Could not get your name."})

    if not (len(pin) == 4 and pin.isdigit()):
        pin = ""
        for _ in range(4):
            bank_core.speak("Please say your 4 digit pin, digit by digit.")
            heard = bank_core.listen(max_wait=12, silence_end=2.0)
            candidate = bank_core.extract_numbers(heard)
            if len(candidate) == 4:
                pin = candidate
                break
            bank_core.speak("That was not 4 digits. Please try again, speaking clearly and a little slower.")
        if not pin:
            bank_core.speak("I could not get a valid pin. Please try again, or type it on the page.")
            return jsonify({"success": False, "message": "Could not get a valid pin."})

    bank_core.speak(f"Creating your account, {name}.")
    account_id = bank_core.create_account(name, pin)
    if account_id is None:
        return jsonify({"success": False, "message": "Account was not created. Face or voice registration failed."})

    bank_core.speak(f"Account created successfully. Your account number is {account_id}. Please remember this number.")
    return jsonify({"success": True, "name": name, "account_id": account_id})


@app.route('/api/login_flow', methods=['POST'])
@exclusive
def login_flow():
    data = request.json or {}
    account_id_raw = (data.get('account_id') or "").strip()
    pin = (data.get('pin') or "").strip()

    if not account_id_raw:
        for _ in range(3):
            bank_core.speak("Please say your account number, digit by digit.")
            heard = bank_core.listen(max_wait=20, silence_end=2.0)
            candidate = bank_core.extract_numbers(heard)
            if candidate:
                account_id_raw = candidate
                break
            if heard:
                bank_core.speak("That did not sound like an account number. Let's try again.")
        if not account_id_raw:
            bank_core.speak("I could not get your account number. Please try again, or type it on the page.")
            return jsonify({"success": False, "message": "Could not get your account number."})

    try:
        account_id = int(account_id_raw)
    except ValueError:
        bank_core.speak("I could not understand the account number. Please try again, or type it on the page.")
        return jsonify({"success": False, "message": "Invalid account number."})

    if not (len(pin) == 4 and pin.isdigit()):
        pin = ""
        for _ in range(4):
            bank_core.speak("Please say your 4 digit pin, digit by digit.")
            heard = bank_core.listen(max_wait=12, silence_end=2.0)
            candidate = bank_core.extract_numbers(heard)
            if len(candidate) == 4:
                pin = candidate
                break
            bank_core.speak("That was not 4 digits. Please try again, speaking clearly and a little slower.")
        if not pin:
            bank_core.speak("I could not get a valid pin. Please try again, or type it on the page.")
            return jsonify({"success": False, "message": "Invalid pin."})

    bank_core.speak("Verifying your PIN and face, please wait.")
    success = bank_core.login(account_id, pin)

    if success:
        bank_core.speak("Login successful. Welcome to VoiceBank.")
        return jsonify({"success": True, "account_id": account_id})

    bank_core.speak("Login failed. Incorrect PIN or face not recognized.")
    return jsonify({"success": False, "message": "Login failed. Incorrect PIN or face not recognized."})


@app.route('/api/cancel_listen', methods=['POST'])
def api_cancel_listen():
    """Not @exclusive on purpose: it must be able to run WHILE the lock is held by an
    in-progress /api/voice_menu call, so a manual button click can interrupt it fast."""
    bank_core.request_cancel_listen()
    return jsonify({"cancelled": True})


@app.route('/api/voice_menu', methods=['POST'])
@exclusive
def api_voice_menu():
    data = request.json or {}
    account_id = int(data.get('account_id'))
    first = bool(data.get('first'))

    if first:
        bank_core.speak("You can say: check balance, deposit, withdraw, mini statement, or logout. What would you like to do?")
    else:
        bank_core.speak("What would you like to do next? Say check balance, deposit, withdraw, statement, or logout.")

    heard = bank_core.listen(max_wait=12, silence_end=1.5)

    if bank_core.cancel_listen_event.is_set():
        bank_core.cancel_listen_event.clear()
        return jsonify({"action": "cancelled", "heard": "", "message": "", "transactions": []})

    action = bank_core.parse_command(heard)
    result = {"action": action, "heard": heard, "message": "", "transactions": []}

    if action == 'balance':
        balance = run_balance(account_id)
        result["message"] = f"Balance: {balance:.2f} rupees"
    elif action == 'deposit':
        result["message"] = run_deposit(account_id)["message"]
    elif action == 'withdraw':
        result["message"] = run_withdraw(account_id)["message"]
    elif action == 'statement':
        result["transactions"] = run_statement(account_id)
        result["message"] = "" if result["transactions"] else "No recent transactions."
    elif action == 'logout':
        bank_core.speak("You have been logged out safely. Goodbye.")
        result["message"] = "Logged out."
    elif action == 'unknown':
        bank_core.speak(f"I heard {heard}, but that is not one of the options.")
        result["message"] = f"Did not understand: {heard}"
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