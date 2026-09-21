"""
VoiceBank - Fully Accessible Voice & Face-Guided Banking Application
Designed for visually impaired users: Speaks instructions out loud and listens to voice commands.
Uses sounddevice for microphone recording and speech_recognition for command processing.
"""

import sqlite3
import hashlib
import hmac
import os
import time
import wave
from datetime import datetime
import cv2
import pyttsx3
import sounddevice as sd
import speech_recognition as sr

DB_NAME = "voicebank.db"
FACES_DIR = "face_data"

if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR)

# ---------- ACCESSIBLE SPEECH OUTPUT & VOICE INPUT ----------

def speak(text):
    """Prints and speaks text clearly through speakers for blind users."""
    print(f"\n[VoiceBank]: {text}")
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 145)  # Optimal accessibility speech rate
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[Speech Output Error]: {e}")

def listen(duration=4):
    """Records speech using sounddevice and processes text with Google Speech Recognition."""
    filename = "temp_voice.wav"
    sample_rate = 44100
    
    print("\n[VoiceBank is listening... Speak now]")
    try:
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(recording.tobytes())

        recognizer = sr.Recognizer()
        with sr.AudioFile(filename) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            print(f"[Recognized Speech]: {text}")
            return text.strip().lower()
            
    except sr.UnknownValueError:
        speak("Sorry, I could not understand what you said.")
    except sr.RequestError:
        speak("Speech recognition service is unreachable.")
    except Exception as e:
        print(f"[Voice Capture Notice]: {e}")
    finally:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass
            
    return ""

def prompt_input(prompt_text, duration=4):
    """Speaks prompt, listens for spoken response, and falls back to terminal typing if silent."""
    speak(prompt_text)
    user_voice = listen(duration=duration)
    
    if user_voice:
        return user_voice
    
    speak("Please type your input in the terminal.")
    return input("> ").strip().lower()

# ---------- SPEECH NUMBER & PIN CONVERTER ----------

def extract_numbers(text, expected_length=None):
    """
    Parses spoken digits, number words, and handles leading zeros.
    Example: 'zero three one nine' -> '0319'
    """
    word_to_num = {
        'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'to': '2', 'too': '2',
        'three': '3', 'four': '4', 'for': '4', 'five': '5', 'six': '6',
        'seven': '7', 'eight': '8', 'ate': '8', 'nine': '9', 'hundred': '00', 'thousand': '000'
    }
    
    words = text.lower().split()
    converted_digits = ""
    
    for w in words:
        if w in word_to_num:
            converted_digits += word_to_num[w]
        else:
            converted_digits += "".join(c for c in w if c.isdigit())
            
    # Auto-pad single missing leading zero for 4-digit PINs (e.g. "319" -> "0319")
    if expected_length and len(converted_digits) == expected_length - 1:
        converted_digits = "0" + converted_digits

    return converted_digits

# ---------- FACE VERIFICATION MODULE ----------

def capture_face_image(save_path):
    """Captures a webcam picture for account registration or verification."""
    speak("Opening camera. Please face the webcam directly.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        speak("Camera unavailable. Proceeding with PIN security only.")
        return False

    time.sleep(1)  # Warm up camera sensor
    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        cv2.imwrite(save_path, frame)
        speak("Facial profile captured successfully.")
        return True
    else:
        speak("Camera snapshot failed.")
        return False

def verify_face(account_id):
    """Verifies user face against registered snapshot using histogram correlation."""
    stored_face_path = os.path.join(FACES_DIR, f"user_{account_id}.jpg")
    
    if not os.path.exists(stored_face_path):
        speak("No registered face profile found. Access granted with PIN.")
        return True

    temp_login_path = "temp_login.jpg"
    captured = capture_face_image(temp_login_path)
    
    if not captured or not os.path.exists(temp_login_path):
        speak("Face check unavailable. PIN access granted.")
        return True

    img1 = cv2.imread(stored_face_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(temp_login_path, cv2.IMREAD_GRAYSCALE)
    
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)

    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    if os.path.exists(temp_login_path):
        os.remove(temp_login_path)

    print(f"Face Match Score: {score:.2f}")
    if score > 0.4:
        speak("Face verification passed.")
        return True
    else:
        speak("Face verification failed. Access denied.")
        return False

# ---------- DATABASE SETUP ----------

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            salt BLOB NOT NULL,
            pin_hash BLOB NOT NULL,
            balance REAL NOT NULL DEFAULT 0
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(id)
        )
    """)
    conn.commit()
    conn.close()

# ---------- BANKING LOGIC ----------

def hash_pin(pin, salt):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 100_000)

def create_account(name, pin, opening_balance=0):
    salt = os.urandom(16)
    pin_hash = hash_pin(pin, salt)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO accounts (name, salt, pin_hash, balance) VALUES (?, ?, ?, ?)",
        (name, salt, pin_hash, opening_balance),
    )
    account_id = cur.lastrowid
    conn.commit()
    conn.close()
    
    face_path = os.path.join(FACES_DIR, f"user_{account_id}.jpg")
    capture_face_image(face_path)
    
    return account_id

def login(account_id, pin):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT salt, pin_hash FROM accounts WHERE id = ?", (account_id,))
    row = cur.fetchone()
    conn.close()
    
    if row is None:
        return False
        
    salt, stored_hash = row
    pin_correct = hmac.compare_digest(hash_pin(pin, salt), stored_hash)
    
    if not pin_correct:
        return False

    return verify_face(account_id)

def get_balance(account_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT balance FROM accounts WHERE id = ?", (account_id,))
    balance = cur.fetchone()[0]
    conn.close()
    return balance

def record_transaction(cur, account_id, kind, amount):
    cur.execute(
        "INSERT INTO transactions (account_id, type, amount, timestamp) VALUES (?, ?, ?, ?)",
        (account_id, kind, amount, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )

def deposit(account_id, amount):
    if amount <= 0:
        return False, "Amount must be greater than zero."
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, account_id))
    record_transaction(cur, account_id, "DEPOSIT", amount)
    conn.commit()
    conn.close()
    return True, f"Successfully deposited {amount:.2f} rupees. Your new balance is {get_balance(account_id):.2f} rupees."

def withdraw(account_id, amount):
    if amount <= 0:
        return False, "Amount must be greater than zero."
    current_bal = get_balance(account_id)
    if amount > current_bal:
        return False, f"Insufficient balance. Your current balance is only {current_bal:.2f} rupees."
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, account_id))
    record_transaction(cur, account_id, "WITHDRAW", amount)
    conn.commit()
    conn.close()
    return True, f"Please collect your cash. Successfully withdrew {amount:.2f} rupees. Remaining balance is {get_balance(account_id):.2f} rupees."

def mini_statement(account_id, limit=5):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT type, amount, timestamp FROM transactions "
        "WHERE account_id = ? ORDER BY id DESC LIMIT ?",
        (account_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return rows

def read_amount_voice(action_type="withdraw"):
    """Asks for transaction amount out loud and returns float."""
    while True:
        speech = prompt_input(f"Please say the amount you want to {action_type}.")
        clean_num_str = extract_numbers(speech)
        try:
            val = float(clean_num_str)
            if val > 0:
                return val
            else:
                speak("Amount must be greater than zero.")
        except ValueError:
            speak("I could not recognize the amount. Please say a number clearly, like 500 or 1000.")

# ---------- ACCESSIBLE ACCOUNT MENU ----------

def account_menu(account_id):
    speak("Login successful. Welcome to your account menu.")
    while True:
        choice = prompt_input("Say 1 or Balance, 2 or Deposit, 3 or Withdraw, 4 for Statement, or 5 to Logout.")
        
        if "1" in choice or "balance" in choice or "one" in choice:
            speak(f"Your current available balance is {get_balance(account_id):.2f} rupees.")
            
        elif "2" in choice or "deposit" in choice or "two" in choice:
            amt = read_amount_voice("deposit")
            success, msg = deposit(account_id, amt)
            speak(msg)
            
        elif "3" in choice or "withdraw" in choice or "three" in choice or "draw" in choice or "money" in choice:
            amt = read_amount_voice("withdraw")
            success, msg = withdraw(account_id, amt)
            speak(msg)
            
        elif "4" in choice or "statement" in choice or "four" in choice or "history" in choice:
            txs = mini_statement(account_id)
            if not txs:
                speak("You have no recent transactions.")
            else:
                speak(f"Here are your last {len(txs)} transactions.")
                for kind, amount, when in txs:
                    speak(f"{kind} of {amount:.2f} rupees on {when}.")
                    
        elif "5" in choice or "logout" in choice or "five" in choice or "exit" in choice:
            speak("You have been logged out safely.")
            break
        else:
            speak("Option not recognized. Please try again.")

def main():
    init_db()
    speak("Welcome to VoiceBank accessible services.")
    while True:
        choice = prompt_input("Say 1 or Create Account, 2 or Login, or 3 to Exit.")
        
        if "1" in choice or "create" in choice or "one" in choice:
            name_speech = prompt_input("Please say your full name.")
            name = name_speech if name_speech else "User"
            
            pin_speech = prompt_input("Please say your 4-digit PIN digit by digit, like zero three one nine.")
            pin = extract_numbers(pin_speech, expected_length=4)
            
            if len(pin) != 4:
                speak("PIN must be exactly 4 digits. Let's try creating the account again.")
                continue
                
            account_id = create_account(name, pin)
            speak(f"Account created successfully for {name}. Your account number is {account_id}. Please remember this number.")
            
        elif "2" in choice or "login" in choice or "two" in choice:
            acc_speech = prompt_input("Please say your account number.")
            acc_clean = extract_numbers(acc_speech)
            
            try:
                account_id = int(acc_clean)
            except ValueError:
                speak("Invalid account number format. Returning to main menu.")
                continue
                
            pin_speech = prompt_input("Please say your 4-digit PIN digit by digit.")
            pin = extract_numbers(pin_speech, expected_length=4)
            
            if login(account_id, pin):
                account_menu(account_id)
            else:
                speak("Authentication failed. Wrong account number, PIN, or facial verification mismatch.")
                
        elif "3" in choice or "exit" in choice or "three" in choice:
            speak("Thank you for using VoiceBank. Have a wonderful day! Goodbye.")
            break
        else:
            speak("Option not recognized. Please try again.")

if __name__ == "__main__":
    main()