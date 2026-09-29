"""
VoiceBank - Core Module
Voice + face guided banking for blind and low-vision users.

WHAT CHANGED IN THIS VERSION (speed fix - speech INPUT only)
------------------------------------------------------------
Problem: the app took ~4 minutes to reply after the user spoke.

Cause: the recorder never detected the END of speech. The speech threshold
(10) was BELOW the real background noise of the mic, so "speech" was detected
instantly and the end-of-speech level (5) was never reached. Every recording
ran to the 30 s limit, and Whisper then decoded 30 s of noise with beam search
and two language passes, hallucinating loops such as "log in, log in, ...".

Fixes:
1. VAD_MIN_THRESHOLD raised to 45 (above typical mic noise).
2. End-of-speech level is now RELATIVE to the speaker's own level
   (END_LEVEL_FRACTION), so background noise cannot keep recording open.
3. Max recording length 10 s (15 s for digits, 8 s for yes/no and voice sample).
4. Audio sent to Whisper is capped at MAX_WHISPER_SECONDS.
5. Whisper uses greedy decoding + no timestamps (several times faster).
6. Repeated-loop hallucinations are collapsed (PIN digit strings untouched).
7. Whisper decode time is printed for each pass.

Everything else (flows, MESSAGES, face, database, server.py, index.html)
works exactly as before.

Install with: pip install gTTS pygame deep-translator noisereduce openai-whisper scipy
"""

import sqlite3
import hashlib
import hmac
import json
import os
import re
import time
import wave
import threading
import unicodedata
from collections import deque
import winsound
from difflib import get_close_matches, SequenceMatcher
from datetime import datetime

import cv2
import numpy as np
import pyttsx3
import sounddevice as sd
import librosa

try:
    import pythoncom
except ImportError:
    pythoncom = None

try:
    from gtts import gTTS
    import pygame
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False
    print("[Setup] gTTS/pygame not installed - multi-language speech output will "
          "fall back to English-only offline voice. Run: pip install gTTS pygame")

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
    _TRANSLATE_AVAILABLE = True
except ImportError:
    _TRANSLATE_AVAILABLE = False

try:
    import noisereduce as nr
    _NOISEREDUCE_AVAILABLE = True
except ImportError:
    _NOISEREDUCE_AVAILABLE = False
    print("[Setup] noisereduce not installed - recordings will be sent to "
          "speech recognition without background-noise cleanup. Run: "
          "pip install noisereduce")

try:
    from scipy.signal import butter, sosfilt
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ======================= SPEECH RECOGNITION (Whisper) =======================
# "small" is a good balance. If decoding is still slow on your CPU, change to
# "base" (2-3x faster, slightly less accurate). Change this one line.
WHISPER_MODEL_SIZE = "small"

try:
    import whisper
    print(f"[Setup] Loading Whisper '{WHISPER_MODEL_SIZE}' model "
          f"(first load can take a little while)...")
    _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
    print("[Setup] Whisper model loaded.")
    _WHISPER_AVAILABLE = True
except ImportError:
    _whisper_model = None
    _WHISPER_AVAILABLE = False
    print("[Setup] openai-whisper not installed - speech recognition will not "
          "work. Run: pip install openai-whisper")
except Exception as e:
    _whisper_model = None
    _WHISPER_AVAILABLE = False
    print(f"[Setup] Whisper failed to load: {e}")

# ======================= SETTINGS (tune here) =======================
DB_NAME = "voicebank.db"
FACES_DIR = "face_data"
VOICES_DIR = "voice_data"
TTS_CACHE_DIR = "tts_cache"

# Microphone / listening
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.05

# --- Voice activity detection ---
CALIBRATION_SKIP_BLOCKS = 2      # ignore first 0.1 s (stream start click / beep tail)
NOISE_WINDOW_BLOCKS = 30         # rolling 1.5 s window used to measure room noise
NOISE_PERCENTILE = 20
VAD_MIN_THRESHOLD = 45           # was 10 - that was BELOW real mic noise, so it "heard speech" instantly
VAD_MAX_THRESHOLD = 400
VAD_NOISE_MULTIPLIER = 3.0
SPEECH_START_BLOCKS = 3          # need 3 loud blocks in a row (0.15 s) to start
END_LEVEL_FRACTION = 0.12        # speech has ENDED when level < 12% of your speaking level
PRE_ROLL_BLOCKS = 10             # keep 0.5 s before speech
POST_ROLL_BLOCKS = 8             # keep 0.4 s after speech
MIN_SPEECH_BLOCKS = 4
DEFAULT_MAX_TOTAL = 10           # was 30 - hard cap on one recording (seconds)

# --- Audio preparation for Whisper ---
WHISPER_TARGET_PEAK = 0.90
WHISPER_MAX_GAIN = 60.0
WHISPER_EDGE_PAD_SECONDS = 0.25
WHISPER_HIGHPASS_HZ = 70
MAX_WHISPER_SECONDS = 12         # never send Whisper more than this

PRE_LISTEN_SETTLE_SECONDS = 0.4

# Camera
CAMERA_INDEX = 0
CAMERA_BACKENDS = [
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_ANY", cv2.CAP_ANY),
]
CAMERA_INDICES_TO_TRY = [0, 1]
CAMERA_WARMUP_READS = 5
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
DETECT_WIDTH = 640
CAMERA_TIMEOUT = 20
GUIDANCE_GAP = 4

# Verification
FACE_MATCH_THRESHOLD = 65
VOICE_MAX_DISTANCE = 40.0
MIN_VOICE_ENERGY = 80            # was 150 - forced people to speak loudly at withdrawal

# Cash dispensing sound
DISPENSE_BEEP_COUNT = 6
DISPENSE_BEEP_MS = 90
DISPENSE_BEEP_FREQS = (750, 1050)

# Fuzzy matching cutoffs
FUZZY_CUTOFF = 0.68
FUZZY_CUTOFF_STRICT = 0.75

for _path in [FACES_DIR, VOICES_DIR, TTS_CACHE_DIR]:
    if not os.path.exists(_path):
        os.makedirs(_path)

frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

speak_lock = threading.Lock()

if _GTTS_AVAILABLE:
    try:
        pygame.mixer.init()
    except Exception as e:
        print(f"[Setup] pygame mixer failed to init: {e}")
        _GTTS_AVAILABLE = False


# ======================= LANGUAGE SUPPORT =======================
LANGUAGES = {
    "english": {
        "label": "English",
        "stt": "en-IN",
        "tts": "en",
        "yes": {"yes", "yeah", "yep", "yup", "confirm", "correct", "right",
                "ok", "okay", "sure", "proceed", "ya", "haan", "affirmative"},
        "no": {"no", "nope", "cancel", "stop", "wrong", "negative",
               "incorrect", "nah", "na"},
        "commands": {
            "logout": ["log out", "logout", "log off", "logoff", "sign out", "sign off",
                       "exit", "quit", "goodbye", "bye", "leave", "finish"],
            "balance": ["balance", "ballance", "check balance", "my balance", "bank balance",
                        "account balance", "how much", "money left", "remaining", "available",
                        "show balance", "total"],
            "withdraw": ["withdraw", "withdrawal", "withdrew", "take out", "take money",
                         "debit", "get money", "get cash", "take cash", "withdraw cash",
                         "cash out", "need money", "remove money", "pull out"],
            "deposit": ["deposit", "depositing", "add money", "put money", "credit",
                        "save money", "add cash", "put cash", "put in", "top up", "topup"],
            "statement": ["statement", "transaction", "transactions", "history", "recent",
                          "mini statement", "ministatement", "last transactions",
                          "passbook", "mini", "recent activity"],
        },
        "mode": {
            "create": ["create account", "create an account", "create a new account",
                       "new account", "open account", "open an account", "sign up",
                       "create", "register", "registration", "new user", "i am new",
                       "first time", "make account", "make an account", "open", "new"],
            "login": ["login", "log in", "log-in", "logon", "sign in", "existing",
                      "already have", "i have an account", "old account", "existing user"],
        },
        "names": ["english", "inglish", "ingleesh", "angrezi", "इंग्लिश", "इंग्लीश", "अंग्रेजी", "अंग्रेज़ी", "ಇಂಗ್ಲಿಷ್", "ఇంగ్లీష్", "ஆங்கிலம்", "இங்கிலீஷ்"],
    },
    "hindi": {
        "label": "Hindi",
        "stt": "hi-IN",
        "tts": "hi",
        "yes": {"हाँ", "हां", "जी", "सही", "ठीक", "ठीक है", "बिल्कुल", "जी हाँ", "han", "haan", "sahi", "theek", "thik"},
        "no": {"नहीं", "गलत", "रुको", "रद्द", "nahi", "nahin", "galat", "ruko"},
        "commands": {
            "logout": ["लॉग आउट", "बाहर", "बंद करो", "बाहर निकलो", "log out", "bahar"],
            "balance": ["बैलेंस", "शेष", "बकाया", "balance"],
            "withdraw": ["निकासी", "पैसे निकालो", "निकालना", "withdraw", "nikasi"],
            "deposit": ["जमा", "पैसे जमा करो", "जमा करना", "deposit", "jama"],
            "statement": ["विवरण", "लेन देन", "स्टेटमेंट", "statement"],
        },
        "mode": {
            "create": ["खाता खोलो", "नया खाता", "अकाउंट बनाओ", "create account", "khata kholo", "naya khata"],
            "login": ["लॉग इन", "लॉगिन", "प्रवेश करो", "login", "log in"],
        },
        "names": ["hindi", "hindhi", "indi", "हिंदी", "हिन्दी", "ಹಿಂದಿ", "హిందీ", "இந்தி"],
    },
    "kannada": {
        "label": "Kannada",
        "stt": "kn-IN",
        "tts": "kn",
        "yes": {"ಹೌದು", "ಸರಿ", "ಸರಿಯಾಗಿದೆ", "howdu", "haudu", "sari"},
        "no": {"ಇಲ್ಲ", "ತಪ್ಪು", "ರದ್ದು", "illa", "tappu"},
        "commands": {
            "logout": ["ಲಾಗ್ ಔಟ್", "ಹೊರಗೆ", "log out"],
            "balance": ["ಬ್ಯಾಲೆನ್ಸ್", "ಶಿಲ್ಕು", "balance"],
            "withdraw": ["ಹಿಂಪಡೆಯುವಿಕೆ", "ಹಣ ತೆಗೆ", "withdraw"],
            "deposit": ["ಠೇವಣಿ", "ಹಣ ಹಾಕು", "deposit"],
            "statement": ["ಹೇಳಿಕೆ", "ವಹಿವಾಟು", "statement"],
        },
        "mode": {
            "create": ["ಖಾತೆ ತೆರೆ", "ಹೊಸ ಖಾತೆ", "create account", "khate tere"],
            "login": ["ಲಾಗಿನ್", "ಪ್ರವೇಶ", "login"],
        },
        "names": ["kannada", "kannad", "kanada", "canada", "कन्नड़", "कन्नड", "कन्नडा", "ಕನ್ನಡ", "కన్నడ", "கன்னடம்"],
    },
    "telugu": {
        "label": "Telugu",
        "stt": "te-IN",
        "tts": "te",
        "yes": {"అవును", "సరే", "సరిగ్గా", "avunu", "sare", "sariga"},
        "no": {"కాదు", "తప్పు", "రద్దు", "kaadu", "kadu", "thappu"},
        "commands": {
            "logout": ["లాగ్ అవుట్", "బయటకు", "log out"],
            "balance": ["బ్యాలెన్స్", "నిల్వ", "balance"],
            "withdraw": ["విత్‌డ్రా", "డబ్బు తీయి", "withdraw"],
            "deposit": ["డిపాజిట్", "డబ్బు వేయి", "deposit"],
            "statement": ["స్టేట్‌మెంట్", "లావాదేవీలు", "statement"],
        },
        "mode": {
            "create": ["ఖాతా తెరవండి", "కొత్త ఖాతా", "create account"],
            "login": ["లాగిన్", "ప్రవేశించు", "login"],
        },
        "names": ["telugu", "telegu", "telgu", "thelugu", "तेलुगु", "तेलुगू", "तेलगु", "तेलगू", "ತೆಲುಗು", "తెలుగు", "தெலுங்கு"],
    },
    "tamil": {
        "label": "Tamil",
        "stt": "ta-IN",
        "tts": "ta",
        "yes": {"ஆம்", "சரி", "சரியாக", "aam", "seri", "sari", "shari"},
        "no": {"இல்லை", "தவறு", "ரத்து", "illai", "thappu", "vendam", "வேண்டாம்"},
        "commands": {
            "logout": ["வெளியேறு", "லாக் அவுட்", "log out"],
            "balance": ["இருப்பு", "பேலன்ஸ்", "balance"],
            "withdraw": ["பணம் எடு", "எடு", "withdraw"],
            "deposit": ["டெபாசிட்", "பணம் போடு", "deposit"],
            "statement": ["அறிக்கை", "பரிவர்த்தனை", "statement"],
        },
        "mode": {
            "create": ["கணக்கு திற", "புதிய கணக்கு", "create account"],
            "login": ["உள்நுழை", "லாகின்", "login"],
        },
        "names": ["tamil", "thamil", "tamizh", "तमिल", "तमिळ", "ತಮಿಳು", "తమిళం", "தமிழ்"],
    },
}

_language_lock = threading.Lock()
_current_language = "english"


def set_language(name):
    global _current_language
    name = (name or "").strip().lower()
    if name not in LANGUAGES:
        return False
    with _language_lock:
        _current_language = name
    return True


def get_language():
    with _language_lock:
        return _current_language


def _lang_conf():
    return LANGUAGES[get_language()]


def available_languages():
    return [(key, conf["label"]) for key, conf in LANGUAGES.items()]


# ======================= PRE-TRANSLATED MESSAGES =======================
MESSAGES = {
    "ask_language": {
        "english": "Which language are you comfortable with? Please say English, Hindi, Kannada, Telugu, or Tamil.",
        "hindi": "आप किस भाषा में सहज हैं? कृपया इंग्लिश, हिंदी, कन्नड़, तेलुगु, या तमिल बोलें।",
        "kannada": "ನೀವು ಯಾವ ಭಾಷೆಯಲ್ಲಿ ಆರಾಮದಾಯಕವಾಗಿದ್ದೀರಿ? ದಯವಿಟ್ಟು ಇಂಗ್ಲಿಷ್, ಹಿಂದಿ, ಕನ್ನಡ, ತೆಲುಗು ಅಥವಾ ತಮಿಳು ಎಂದು ಹೇಳಿ.",
        "telugu": "మీకు ఏ భాష సౌకర్యంగా ఉంటుంది? దయచేసి ఇంగ్లీష్, హిందీ, కన్నడ, తెలుగు, లేదా తమిళం అని చెప్పండి.",
        "tamil": "நீங்கள் எந்த மொழியில் வசதியாக இருக்கிறீர்கள்? தயவுசெய்து ஆங்கிலம், இந்தி, கன்னடம், தெலுங்கு, அல்லது தமிழ் என்று சொல்லுங்கள்.",
    },
    "language_not_understood": {
        "english": "Sorry, I did not catch that. You can also tap your language at the top of the screen.",
        "hindi": "माफ़ कीजिए, मुझे समझ नहीं आया। आप स्क्रीन के ऊपर अपनी भाषा भी चुन सकते हैं।",
        "kannada": "ಕ್ಷಮಿಸಿ, ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ನೀವು ಪರದೆಯ ಮೇಲ್ಭಾಗದಲ್ಲಿ ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಸಹ ಆಯ್ಕೆ ಮಾಡಬಹುದು.",
        "telugu": "క్షమించండి, నాకు అర్థం కాలేదు. మీరు స్క్రీన్ పైభాగంలో మీ భాషను కూడా ఎంచుకోవచ్చు.",
        "tamil": "மன்னிக்கவும், எனக்கு புரியவில்லை. நீங்கள் திரையின் மேற்பகுதியில் உங்கள் மொழியையும் தேர்ந்தெடுக்கலாம்.",
    },
    "language_set": {
        "english": "Language set to English.",
        "hindi": "भाषा हिंदी में सेट कर दी गई है।",
        "kannada": "ಭಾಷೆ ಕನ್ನಡಕ್ಕೆ ಹೊಂದಿಸಲಾಗಿದೆ.",
        "telugu": "భాష తెలుగుకు సెట్ చేయబడింది.",
        "tamil": "மொழி தமிழுக்கு அமைக்கப்பட்டது.",
    },
    "ask_mode": {
        "english": "Say create account to open a new account, or say login if you already have one.",
        "hindi": "नया खाता खोलने के लिए 'खाता खोलें' बोलें, या अगर आपका पहले से खाता है तो 'लॉगिन' बोलें।",
        "kannada": "ಹೊಸ ಖಾತೆ ತೆರೆಯಲು 'ಖಾತೆ ತೆರೆ' ಎಂದು ಹೇಳಿ, ಅಥವಾ ನಿಮ್ಮಲ್ಲಿ ಈಗಾಗಲೇ ಖಾತೆ ಇದ್ದರೆ 'ಲಾಗಿನ್' ಎಂದು ಹೇಳಿ.",
        "telugu": "కొత్త ఖాతా తెరవడానికి 'ఖాతా తెరవండి' అని చెప్పండి, లేదా మీకు ఇప్పటికే ఖాతా ఉంటే 'లాగిన్' అని చెప్పండి.",
        "tamil": "புதிய கணக்கு திறக்க 'கணக்கு திற' என்று சொல்லுங்கள், அல்லது ஏற்கனவே கணக்கு இருந்தால் 'உள்நுழை' என்று சொல்லுங்கள்.",
    },
    "mode_not_understood": {
        "english": "I did not catch that. You can also press Create Account or Login on the screen.",
        "hindi": "मुझे समझ नहीं आया। आप स्क्रीन पर 'खाता बनाएं' या 'लॉगिन' भी दबा सकते हैं।",
        "kannada": "ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ. ನೀವು ಪರದೆಯ ಮೇಲೆ 'ಖಾತೆ ರಚಿಸಿ' ಅಥವಾ 'ಲಾಗಿನ್' ಸಹ ಒತ್ತಬಹುದು.",
        "telugu": "నాకు అర్థం కాలేదు. మీరు స్క్రీన్‌పై 'ఖాతా సృష్టించు' లేదా 'లాగిన్' కూడా నొక్కవచ్చు.",
        "tamil": "எனக்கு புரியவில்லை. நீங்கள் திரையில் 'கணக்கை உருவாக்கு' அல்லது 'உள்நுழை' என்பதையும் அழுத்தலாம்.",
    },
    "ask_name": {
        "english": "Please say your full name.",
        "hindi": "कृपया अपना पूरा नाम बोलें।",
        "kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪೂರ್ಣ ಹೆಸರನ್ನು ಹೇಳಿ.",
        "telugu": "దయచేసి మీ పూర్తి పేరు చెప్పండి.",
        "tamil": "தயவுசெய்து உங்கள் முழுப் பெயரைச் சொல்லுங்கள்.",
    },
    "could_not_get_name": {
        "english": "I could not get your name. Please try again.",
        "hindi": "मुझे आपका नाम नहीं मिल सका। कृपया फिर से प्रयास करें।",
        "kannada": "ನನಗೆ ನಿಮ್ಮ ಹೆಸರು ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "telugu": "నాకు మీ పేరు దొరకలేదు. దయచేసి మళ్లీ ప్రయత్నించండి.",
        "tamil": "எனக்கு உங்கள் பெயர் கிடைக்கவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
    },
    "ask_pin": {
        "english": "Please say your 4 digit pin, digit by digit.",
        "hindi": "कृपया अपना 4 अंकों का पिन, एक-एक अंक करके बोलें।",
        "kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ 4 ಅಂಕಿಗಳ ಪಿನ್ ಅನ್ನು, ಒಂದೊಂದಾಗಿ ಹೇಳಿ.",
        "telugu": "దయచేసి మీ 4 అంకెల పిన్‌ను, ఒక్కొక్క అంకె చెప్పండి.",
        "tamil": "தயவுசெய்து உங்கள் 4 இலக்க பின்னை, ஒவ்வொரு இலக்கமாகச் சொல்லுங்கள்.",
    },
    "could_not_get_pin": {
        "english": "I could not get a valid pin. Please try again, or type it on the page.",
        "hindi": "मुझे एक मान्य पिन नहीं मिल सका। कृपया फिर से प्रयास करें, या इसे पेज पर टाइप करें।",
        "kannada": "ನನಗೆ ಮಾನ್ಯವಾದ ಪಿನ್ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ, ಅಥವಾ ಪುಟದಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ.",
        "telugu": "నాకు చెల్లుబాటు అయ్యే పిన్ దొరకలేదు. దయచేసి మళ్లీ ప్రయత్నించండి, లేదా పేజీలో టైప్ చేయండి.",
        "tamil": "எனக்கு சரியான பின் கிடைக்கவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும், அல்லது பக்கத்தில் தட்டச்சு செய்யவும்.",
    },
    "ask_account_number": {
        "english": "Please say your account number, digit by digit.",
        "hindi": "कृपया अपनी खाता संख्या, एक-एक अंक करके बोलें।",
        "kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಖಾತೆ ಸಂಖ್ಯೆಯನ್ನು, ಒಂದೊಂದಾಗಿ ಹೇಳಿ.",
        "telugu": "దయచేసి మీ ఖాతా నంబర్‌ను, ఒక్కొక్క అంకె చెప్పండి.",
        "tamil": "தயவுசெய்து உங்கள் கணக்கு எண்ணை, ஒவ்வொரு இலக்கமாகச் சொல்லுங்கள்.",
    },
    "could_not_get_account_number": {
        "english": "I could not get your account number. Please try again, or type it on the page.",
        "hindi": "मुझे आपका खाता नंबर नहीं मिल सका। कृपया फिर से प्रयास करें, या इसे पेज पर टाइप करें।",
        "kannada": "ನನಗೆ ನಿಮ್ಮ ಖಾತೆ ಸಂಖ್ಯೆ ಸಿಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ, ಅಥವಾ ಪುಟದಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ.",
        "telugu": "నాకు మీ ఖాతా నంబర్ దొరకలేదు. దయచేసి మళ్లీ ప్రయత్నించండి, లేదా పేజీలో టైప్ చేయండి.",
        "tamil": "எனக்கு உங்கள் கணக்கு எண் கிடைக்கவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும், அல்லது பக்கத்தில் தட்டச்சு செய்யவும்.",
    },
    "digits_not_clear": {
        "english": "That was not clear. Please try again, speaking clearly and a little slower.",
        "hindi": "वह स्पष्ट नहीं था। कृपया फिर से कोशिश करें, साफ़ और थोड़ा धीरे बोलें।",
        "kannada": "ಅದು ಸ್ಪಷ್ಟವಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ, ಸ್ಪಷ್ಟವಾಗಿ ಮತ್ತು ಸ್ವಲ್ಪ ನಿಧಾನವಾಗಿ ಮಾತನಾಡಿ.",
        "telugu": "అది స్పష్టంగా లేదు. దయచేసి మళ్ళీ ప్రయత్నించండి, స్పష్టంగా మరియు కొంచెం నెమ్మదిగా మాట్లాడండి.",
        "tamil": "அது தெளிவாக இல்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும், தெளிவாகவும் சற்று மெதுவாகவும் பேசுங்கள்.",
    },
    "verifying": {
        "english": "Verifying your PIN and face, please wait.",
        "hindi": "आपका पिन और चेहरा जांचा जा रहा है, कृपया प्रतीक्षा करें।",
        "kannada": "ನಿಮ್ಮ ಪಿನ್ ಮತ್ತು ಮುಖವನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ, ದಯವಿಟ್ಟು ಕಾಯಿರಿ.",
        "telugu": "మీ పిన్ మరియు ముఖం ధృవీకరించబడుతోంది, దయచేసి వేచి ఉండండి.",
        "tamil": "உங்கள் பின் மற்றும் முகம் சரிபார்க்கப்படுகிறது, தயவுசெய்து காத்திருங்கள்.",
    },
    "login_success": {
        "english": "Login successful. Welcome to VoiceBank.",
        "hindi": "लॉगिन सफल रहा। VoiceBank में आपका स्वागत है।",
        "kannada": "ಲಾಗಿನ್ ಯಶಸ್ವಿಯಾಗಿದೆ. VoiceBank ಗೆ ಸುಸ್ವಾಗತ.",
        "telugu": "లాగిన్ విజయవంతమైంది. VoiceBankకి స్వాగతం.",
        "tamil": "உள்நுழைவு வெற்றி பெற்றது. VoiceBank-க்கு வரவேற்கிறோம்.",
    },
    "login_failed": {
        "english": "Login failed. Incorrect PIN or face not recognized.",
        "hindi": "लॉगिन विफल रहा। गलत पिन या चेहरा पहचान में नहीं आया।",
        "kannada": "ಲಾಗಿನ್ ವಿಫಲವಾಗಿದೆ. ತಪ್ಪಾದ ಪಿನ್ ಅಥವಾ ಮುಖ ಗುರುತಿಸಲಾಗಲಿಲ್ಲ.",
        "telugu": "లాగిన్ విఫలమైంది. తప్పు పిన్ లేదా ముఖం గుర్తించబడలేదు.",
        "tamil": "உள்நுழைவு தோல்வியடைந்தது. தவறான பின் அல்லது முகம் அடையாளம் காணப்படவில்லை.",
    },
    "menu_first": {
        "english": "You can say: check balance, deposit, withdraw, mini statement, or logout. What would you like to do?",
        "hindi": "आप कह सकते हैं: बैलेंस चेक करो, जमा करो, निकासी करो, मिनी स्टेटमेंट, या लॉग आउट। आप क्या करना चाहेंगे?",
        "kannada": "ನೀವು ಹೇಳಬಹುದು: ಬ್ಯಾಲೆನ್ಸ್ ಪರಿಶೀಲಿಸಿ, ಠೇವಣಿ ಮಾಡಿ, ಹಿಂಪಡೆಯಿರಿ, ಮಿನಿ ಸ್ಟೇಟ್‌ಮೆಂಟ್, ಅಥವಾ ಲಾಗ್ ಔಟ್. ನೀವು ಏನು ಮಾಡಲು ಬಯಸುವಿರಿ?",
        "telugu": "మీరు చెప్పవచ్చు: బ్యాలెన్స్ చెక్ చేయండి, డిపాజిట్ చేయండి, విత్‌డ్రా చేయండి, మినీ స్టేట్‌మెంట్, లేదా లాగ్ అవుట్. మీరు ఏమి చేయాలనుకుంటున్నారు?",
        "tamil": "நீங்கள் சொல்லலாம்: இருப்பு சரிபார், டெபாசிட், பணம் எடு, மினி அறிக்கை, அல்லது லாக் அவுட். நீங்கள் என்ன செய்ய விரும்புகிறீர்கள்?",
    },
    "menu_again": {
        "english": "What would you like to do next? Say check balance, deposit, withdraw, statement, or logout.",
        "hindi": "आगे आप क्या करना चाहेंगे? बैलेंस चेक करो, जमा करो, निकासी करो, स्टेटमेंट, या लॉग आउट बोलें।",
        "kannada": "ಮುಂದೆ ನೀವು ಏನು ಮಾಡಲು ಬಯಸುವಿರಿ? ಬ್ಯಾಲೆನ್ಸ್ ಪರಿಶೀಲಿಸಿ, ಠೇವಣಿ, ಹಿಂಪಡೆಯುವಿಕೆ, ಸ್ಟೇಟ್‌ಮೆಂಟ್, ಅಥವಾ ಲಾಗ್ ಔಟ್ ಎಂದು ಹೇಳಿ.",
        "telugu": "తర్వాత మీరు ఏమి చేయాలనుకుంటున్నారు? బ్యాలెన్స్ చెక్ చేయండి, డిపాజిట్, విత్‌డ్రా, స్టేట్‌మెంట్, లేదా లాగ్ అవుట్ అని చెప్పండి.",
        "tamil": "அடுத்து நீங்கள் என்ன செய்ய விரும்புகிறீர்கள்? இருப்பு சரிபார், டெபாசிட், பணம் எடு, அறிக்கை, அல்லது லாக் அவுட் என்று சொல்லுங்கள்.",
    },
    "please_repeat_command": {
        "english": "I did not quite catch that. Please say it again.",
        "hindi": "मुझे ठीक से समझ नहीं आया। कृपया फिर से कहें।",
        "kannada": "ನನಗೆ ಸರಿಯಾಗಿ ಅರ್ಥವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಹೇಳಿ.",
        "telugu": "నాకు సరిగ్గా అర్థం కాలేదు. దయచేసి మళ్లీ చెప్పండి.",
        "tamil": "எனக்கு சரியாகப் புரியவில்லை. தயவுசெய்து மீண்டும் சொல்லுங்கள்.",
    },
    "unknown_command": {
        "english": "I heard {heard}, but that is not one of the options.",
        "hindi": "मैंने सुना {heard}, लेकिन यह विकल्पों में से एक नहीं है।",
        "kannada": "ನಾನು {heard} ಎಂದು ಕೇಳಿದೆ, ಆದರೆ ಅದು ಆಯ್ಕೆಗಳಲ್ಲಿ ಒಂದಲ್ಲ.",
        "telugu": "నేను {heard} అని విన్నాను, కానీ అది ఎంపికలలో ఒకటి కాదు.",
        "tamil": "நான் {heard} என்று கேட்டேன், ஆனால் அது விருப்பங்களில் ஒன்றல்ல.",
    },
    "logout_command": {
        "english": "You have been logged out safely. Goodbye.",
        "hindi": "आपको सुरक्षित रूप से लॉग आउट कर दिया गया है। अलविदा।",
        "kannada": "ನಿಮ್ಮನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಲಾಗ್ ಔಟ್ ಮಾಡಲಾಗಿದೆ. ವಿದಾಯ.",
        "telugu": "మిమ్మల్ని సురక్షితంగా లాగ్ అవుట్ చేయడం జరిగింది. వీడ్కోలు.",
        "tamil": "நீங்கள் பாதுகாப்பாக லாக் அவுட் செய்யப்பட்டுள்ளீர்கள். போய் வாருங்கள்.",
    },
    "logout_manual": {
        "english": "You have been logged out safely.",
        "hindi": "आपको सुरक्षित रूप से लॉग आउट कर दिया गया है।",
        "kannada": "ನಿಮ್ಮನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಲಾಗ್ ಔಟ್ ಮಾಡಲಾಗಿದೆ.",
        "telugu": "మిమ్మల్ని సురక్షితంగా లాగ్ అవుట్ చేయడం జరిగింది.",
        "tamil": "நீங்கள் பாதுகாப்பாக லாக் அவுட் செய்யப்பட்டுள்ளீர்கள்.",
    },
    "logout_no_response": {
        "english": "No response was heard, so for your security you have been logged out.",
        "hindi": "कोई प्रतिक्रिया नहीं मिली, इसलिए आपकी सुरक्षा के लिए आपको लॉग आउट कर दिया गया है।",
        "kannada": "ಯಾವುದೇ ಪ್ರತಿಕ್ರಿಯೆ ಕೇಳಿಸಲಿಲ್ಲ, ಆದ್ದರಿಂದ ನಿಮ್ಮ ಭದ್ರತೆಗಾಗಿ ನಿಮ್ಮನ್ನು ಲಾಗ್ ಔಟ್ ಮಾಡಲಾಗಿದೆ.",
        "telugu": "ఎటువంటి స్పందన వినిపించలేదు, కాబట్టి మీ భద్రత కోసం మిమ్మల్ని లాగ్ అవుట్ చేయడం జరిగింది.",
        "tamil": "எந்த பதிலும் கேட்கவில்லை, எனவே உங்கள் பாதுகாப்புக்காக நீங்கள் லாக் அவுட் செய்யப்பட்டுள்ளீர்கள்.",
    },
    "nothing_heard": {
        "english": "I did not hear anything.",
        "hindi": "मुझे कुछ भी सुनाई नहीं दिया।",
        "kannada": "ನನಗೆ ಏನೂ ಕೇಳಿಸಲಿಲ್ಲ.",
        "telugu": "నాకు ఏమీ వినిపించలేదు.",
        "tamil": "எனக்கு எதுவும் கேட்கவில்லை.",
    },
    "not_understood": {
        "english": "Sorry, I could not understand what you said.",
        "hindi": "क्षमा करें, मुझे समझ नहीं आया कि आपने क्या कहा।",
        "kannada": "ಕ್ಷಮಿಸಿ, ನೀವು ಏನು ಹೇಳಿದಿರಿ ಎಂದು ನನಗೆ ಅರ್ಥವಾಗಲಿಲ್ಲ.",
        "telugu": "క్షమించండి, మీరు ఏమి చెప్పారో నాకు అర్థం కాలేదు.",
        "tamil": "மன்னிக்கவும், நீங்கள் என்ன சொன்னீர்கள் என்று எனக்குப் புரியவில்லை.",
    },
    "please_confirm_retry": {
        "english": "Please say correct, or wrong.",
        "hindi": "कृपया 'सही' या 'गलत' बोलें।",
        "kannada": "ದಯವಿಟ್ಟು 'ಸರಿ' ಅಥವಾ 'ತಪ್ಪು' ಎಂದು ಹೇಳಿ.",
        "telugu": "దయచేసి 'సరే' లేదా 'తప్పు' అని చెప్పండి.",
        "tamil": "தயவுசெய்து 'சரி' அல்லது 'தவறு' என்று சொல்லுங்கள்.",
    },
    "ask_deposit_amount": {
        "english": "How much would you like to deposit? Please say the amount in rupees.",
        "hindi": "आप कितना जमा करना चाहेंगे? कृपया राशि रुपये में बताएं।",
        "kannada": "ನೀವು ಎಷ್ಟು ಠೇವಣಿ ಮಾಡಲು ಬಯಸುವಿರಿ? ದಯವಿಟ್ಟು ಮೊತ್ತವನ್ನು ರೂಪಾಯಿಗಳಲ್ಲಿ ಹೇಳಿ.",
        "telugu": "మీరు ఎంత డిపాజిట్ చేయాలనుకుంటున్నారు? దయచేసి మొత్తాన్ని రూపాయలలో చెప్పండి.",
        "tamil": "நீங்கள் எவ்வளவு டெபாசிட் செய்ய விரும்புகிறீர்கள்? தயவுசெய்து தொகையை ரூபாயில் சொல்லுங்கள்.",
    },
    "ask_withdraw_amount": {
        "english": "How much would you like to withdraw? Please say the amount in rupees.",
        "hindi": "आप कितना निकालना चाहेंगे? कृपया राशि रुपये में बताएं।",
        "kannada": "ನೀವು ಎಷ್ಟು ಹಿಂಪಡೆಯಲು ಬಯಸುವಿರಿ? ದಯವಿಟ್ಟು ಮೊತ್ತವನ್ನು ರೂಪಾಯಿಗಳಲ್ಲಿ ಹೇಳಿ.",
        "telugu": "మీరు ఎంత విత్‌డ్రా చేయాలనుకుంటున్నారు? దయచేసి మొత్తాన్ని రూపాయలలో చెప్పండి.",
        "tamil": "நீங்கள் எவ்வளவு பணம் எடுக்க விரும்புகிறீர்கள்? தயவுசெய்து தொகையை ரூபாயில் சொல்லுங்கள்.",
    },
    "amount_confirm": {
        "english": "You said {amount} rupees. Say yes to confirm, or say it again to correct it.",
        "hindi": "आपने {amount} रुपये कहा। पुष्टि के लिए हाँ बोलें, या सही करने के लिए फिर से बोलें।",
        "kannada": "ನೀವು {amount} ರೂಪಾಯಿ ಎಂದು ಹೇಳಿದ್ದೀರಿ. ದೃಢೀಕರಿಸಲು ಹೌದು ಎಂದು ಹೇಳಿ, ಅಥವಾ ಸರಿಪಡಿಸಲು ಮತ್ತೆ ಹೇಳಿ.",
        "telugu": "మీరు {amount} రూపాయలు అన్నారు. నిర్ధారించడానికి అవును అని చెప్పండి, లేదా సరిచేయడానికి మళ్లీ చెప్పండి.",
        "tamil": "நீங்கள் {amount} ரூபாய் என்று சொன்னீர்கள். உறுதிப்படுத்த ஆம் என்று சொல்லுங்கள், அல்லது சரிசெய்ய மீண்டும் சொல்லுங்கள்.",
    },
    "amount_not_understood": {
        "english": "I could not understand the amount.",
        "hindi": "मुझे राशि समझ नहीं आई।",
        "kannada": "ನನಗೆ ಮೊತ್ತ ಅರ್ಥವಾಗಲಿಲ್ಲ.",
        "telugu": "నాకు మొత్తం అర్థం కాలేదు.",
        "tamil": "எனக்கு தொகை புரியவில்லை.",
    },
    "deposit_success": {
        "english": "Successfully deposited {amount} rupees. Your new balance is {balance} rupees.",
        "hindi": "सफलतापूर्वक {amount} रुपये जमा किए गए। आपका नया शेष {balance} रुपये है।",
        "kannada": "{amount} ರೂಪಾಯಿಗಳನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಠೇವಣಿ ಮಾಡಲಾಗಿದೆ. ನಿಮ್ಮ ಹೊಸ ಬಾಕಿ {balance} ರೂಪಾಯಿಗಳು.",
        "telugu": "{amount} రూపాయలు విజయవంతంగా జమ చేయబడ్డాయి. మీ కొత్త బ్యాలెన్స్ {balance} రూపాయలు.",
        "tamil": "{amount} ரூபாய் வெற்றிகரமாக டெபாசிட் செய்யப்பட்டது. உங்கள் புதிய இருப்பு {balance} ரூபாய்.",
    },
    "withdraw_success": {
        "english": "Please collect your cash. Successfully withdrew {amount} rupees. Remaining balance is {balance} rupees.",
        "hindi": "कृपया अपना नकद लें। सफलतापूर्वक {amount} रुपये निकाले गए। शेष राशि {balance} रुपये है।",
        "kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ನಗದನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ. {amount} ರೂಪಾಯಿಗಳನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಹಿಂಪಡೆಯಲಾಗಿದೆ. ಉಳಿದ ಬಾಕಿ {balance} ರೂಪಾಯಿಗಳು.",
        "telugu": "దయచేసి మీ నగదును తీసుకోండి. {amount} రూపాయలు విజయవంతంగా విత్‌డ్రా చేయబడ్డాయి. మిగిలిన బ్యాలెన్స్ {balance} రూపాయలు.",
        "tamil": "தயவுசெய்து உங்கள் பணத்தை எடுத்துக் கொள்ளுங்கள். {amount} ரூபாய் வெற்றிகரமாக எடுக்கப்பட்டது. மீதமுள்ள இருப்பு {balance} ரூபாய்.",
    },
    "insufficient_balance": {
        "english": "Insufficient balance. Your current balance is only {balance} rupees.",
        "hindi": "अपर्याप्त शेष। आपका वर्तमान शेष केवल {balance} रुपये है।",
        "kannada": "ಸಾಕಷ್ಟು ಬಾಕಿ ಇಲ್ಲ. ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಬಾಕಿ ಕೇವಲ {balance} ರೂಪಾಯಿಗಳು.",
        "telugu": "సరిపడా బ్యాలెన్స్ లేదు. మీ ప్రస్తుత బ్యాలెన్స్ కేవలం {balance} రూపాయలు మాత్రమే.",
        "tamil": "போதுமான இருப்பு இல்லை. உங்கள் தற்போதைய இருப்பு {balance} ரூபாய் மட்டுமே.",
    },
    "invalid_amount": {
        "english": "Amount must be greater than zero.",
        "hindi": "राशि शून्य से अधिक होनी चाहिए।",
        "kannada": "ಮೊತ್ತ ಶೂನ್ಯಕ್ಕಿಂತ ಹೆಚ್ಚಿರಬೇಕು.",
        "telugu": "మొత్తం సున్నా కంటే ఎక్కువగా ఉండాలి.",
        "tamil": "தொகை பூஜ்ஜியத்தை விட அதிகமாக இருக்க வேண்டும்.",
    },
    "deposit_cancelled": {
        "english": "Deposit cancelled.",
        "hindi": "जमा रद्द कर दिया गया।",
        "kannada": "ಠೇವಣಿ ರದ್ದುಗೊಳಿಸಲಾಗಿದೆ.",
        "telugu": "డిపాజిట్ రద్దు చేయబడింది.",
        "tamil": "டெபாசிட் ரத்து செய்யப்பட்டது.",
    },
    "withdraw_cancelled": {
        "english": "Withdrawal cancelled.",
        "hindi": "निकासी रद्द कर दी गई।",
        "kannada": "ಹಿಂಪಡೆಯುವಿಕೆ ರದ್ದುಗೊಳಿಸಲಾಗಿದೆ.",
        "telugu": "విత్‌డ్రా రద్దు చేయబడింది.",
        "tamil": "பணம் எடுப்பது ரத்து செய்யப்பட்டது.",
    },
    "balance_readout": {
        "english": "Your current available balance is {amount} rupees.",
        "hindi": "आपका वर्तमान उपलब्ध शेष {amount} रुपये है।",
        "kannada": "ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಲಭ್ಯವಿರುವ ಬಾಕಿ {amount} ರೂಪಾಯಿಗಳು.",
        "telugu": "మీ ప్రస్తుత అందుబాటులో ఉన్న బ్యాలెన్స్ {amount} రూపాయలు.",
        "tamil": "உங்கள் தற்போதைய இருப்பு {amount} ரூபாய்.",
    },
    "statement_intro": {
        "english": "Here are your last {count} transactions.",
        "hindi": "यहाँ आपके अंतिम {count} लेन-देन हैं।",
        "kannada": "ಇಲ್ಲಿ ನಿಮ್ಮ ಕೊನೆಯ {count} ವಹಿವಾಟುಗಳಿವೆ.",
        "telugu": "ఇక్కడ మీ చివరి {count} లావాదేవీలు ఉన్నాయి.",
        "tamil": "இதோ உங்கள் கடைசி {count} பரிவர்த்தனைகள்.",
    },
    "statement_line": {
        "english": "{kind} of {amount} rupees on {when}.",
        "hindi": "{when} को {amount} रुपये की {kind}।",
        "kannada": "{when} ರಂದು {amount} ರೂಪಾಯಿಗಳ {kind}.",
        "telugu": "{when} న {amount} రూపాయల {kind}.",
        "tamil": "{when} அன்று {amount} ரூபாய் {kind}.",
    },
    "no_transactions": {
        "english": "You have no recent transactions.",
        "hindi": "आपका कोई हालिया लेन-देन नहीं है।",
        "kannada": "ನಿಮಗೆ ಇತ್ತೀಚಿನ ಯಾವುದೇ ವಹಿವಾಟುಗಳಿಲ್ಲ.",
        "telugu": "మీకు ఇటీవలి లావాదేవీలు ఏవీ లేవు.",
        "tamil": "உங்களுக்கு சமீபத்திய பரிவர்த்தனைகள் இல்லை.",
    },
    "withdrawal_phrase_prompt": {
        "english": "For withdrawal security, after the beep, please say clearly: my voice is my password.",
        "hindi": "निकासी सुरक्षा के लिए, बीप के बाद, कृपया स्पष्ट रूप से बोलें: मेरी आवाज़ ही मेरा पासवर्ड है।",
        "kannada": "ಹಿಂಪಡೆಯುವಿಕೆ ಭದ್ರತೆಗಾಗಿ, ಬೀಪ್ ನಂತರ, ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾಗಿ ಹೇಳಿ: ನನ್ನ ಧ್ವನಿಯೇ ನನ್ನ ಪಾಸ್‌ವರ್ಡ್.",
        "telugu": "విత్‌డ్రా భద్రత కోసం, బీప్ తర్వాత, దయచేసి స్పష్టంగా చెప్పండి: నా వాయిస్ నా పాస్‌వర్డ్.",
        "tamil": "பணம் எடுப்பு பாதுகாப்புக்காக, பீப் ஒலிக்குப் பிறகு, தயவுசெய்து தெளிவாகச் சொல்லுங்கள்: என் குரலே என் கடவுச்சொல்.",
    },
    "voice_verified": {
        "english": "Voice verified.",
        "hindi": "आवाज़ सत्यापित हुई।",
        "kannada": "ಧ್ವನಿ ಪರಿಶೀಲಿಸಲಾಗಿದೆ.",
        "telugu": "వాయిస్ ధృవీకరించబడింది.",
        "tamil": "குரல் சரிபார்க்கப்பட்டது.",
    },
    "voice_not_heard_louder": {
        "english": "I did not hear you. Please speak a little louder and closer to the microphone.",
        "hindi": "मुझे आपकी आवाज़ सुनाई नहीं दी। कृपया थोड़ा जोर से और माइक्रोफ़ोन के करीब बोलें।",
        "kannada": "ನನಗೆ ನಿಮ್ಮ ಧ್ವನಿ ಕೇಳಿಸಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಜೋರಾಗಿ ಮತ್ತು ಮೈಕ್ರೊಫೋನ್‌ಗೆ ಹತ್ತಿರವಾಗಿ ಮಾತನಾಡಿ.",
        "telugu": "నాకు మీ వాయిస్ వినిపించలేదు. దయచేసి కొంచెం గట్టిగా మరియు మైక్రోఫోన్‌కు దగ్గరగా మాట్లాడండి.",
        "tamil": "எனக்கு உங்கள் குரல் கேட்கவில்லை. தயவுசெய்து கொஞ்சம் சத்தமாகவும் மைக்ரோஃபோனுக்கு நெருக்கமாகவும் பேசுங்கள்.",
    },
    "voice_too_quiet": {
        "english": "That was too quiet for me to verify reliably. Please speak a bit louder.",
        "hindi": "यह सत्यापित करने के लिए बहुत धीमा था। कृपया थोड़ा और जोर से बोलें।",
        "kannada": "ವಿಶ್ವಾಸಾರ್ಹವಾಗಿ ಪರಿಶೀಲಿಸಲು ಅದು ತುಂಬಾ ಮೃದುವಾಗಿತ್ತು. ದಯವಿಟ್ಟು ಸ್ವಲ್ಪ ಜೋರಾಗಿ ಮಾತನಾಡಿ.",
        "telugu": "నమ్మదగినంతగా ధృవీకరించడానికి అది చాలా నెమ్మదిగా ఉంది. దయచేసి కొంచెం గట్టిగా మాట్లాడండి.",
        "tamil": "நம்பகமாக சரிபார்க்க அது மிகவும் மெதுவாக இருந்தது. தயவுசெய்து கொஞ்சம் சத்தமாக பேசுங்கள்.",
    },
    "recording_too_short": {
        "english": "That recording was too short or unclear. Please try again.",
        "hindi": "वह रिकॉर्डिंग बहुत छोटी या अस्पष्ट थी। कृपया फिर से प्रयास करें।",
        "kannada": "ಆ ರೆಕಾರ್ಡಿಂಗ್ ತುಂಬಾ ಚಿಕ್ಕದಾಗಿತ್ತು ಅಥವಾ ಅಸ್ಪಷ್ಟವಾಗಿತ್ತು. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "telugu": "ఆ రికార్డింగ్ చాలా చిన్నదిగా లేదా అస్పష్టంగా ఉంది. దయచేసి మళ్లీ ప్రయత్నించండి.",
        "tamil": "அந்த பதிவு மிகவும் குறுகியதாகவோ தெளிவற்றதாகவோ இருந்தது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
    },
    "voice_mismatch_retry": {
        "english": "Your voice did not match closely enough. Let's try once more, speak clearly and steadily.",
        "hindi": "आपकी आवाज़ पर्याप्त रूप से मेल नहीं खाई। एक बार फिर कोशिश करते हैं, स्पष्ट और स्थिर रूप से बोलें।",
        "kannada": "ನಿಮ್ಮ ಧ್ವನಿ ಸಾಕಷ್ಟು ಹೊಂದಿಕೆಯಾಗಲಿಲ್ಲ. ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸೋಣ, ಸ್ಪಷ್ಟವಾಗಿ ಮತ್ತು ಸ್ಥಿರವಾಗಿ ಮಾತನಾಡಿ.",
        "telugu": "మీ వాయిస్ సరిపోలేదు. మరోసారి ప్రయత్నిద్దాం, స్పష్టంగా మరియు స్థిరంగా మాట్లాడండి.",
        "tamil": "உங்கள் குரல் போதுமான அளவு பொருந்தவில்லை. மீண்டும் ஒருமுறை முயற்சிப்போம், தெளிவாகவும் நிலையாகவும் பேசுங்கள்.",
    },
    "voice_verification_failed": {
        "english": "Voice verification failed. Withdrawal cancelled.",
        "hindi": "आवाज़ सत्यापन विफल रहा। निकासी रद्द कर दी गई।",
        "kannada": "ಧ್ವನಿ ಪರಿಶೀಲನೆ ವಿಫಲವಾಗಿದೆ. ಹಿಂಪಡೆಯುವಿಕೆ ರದ್ದುಗೊಳಿಸಲಾಗಿದೆ.",
        "telugu": "వాయిస్ ధృవీకరణ విఫలమైంది. విత్‌డ్రా రద్దు చేయబడింది.",
        "tamil": "குரல் சரிபார்ப்பு தோல்வியடைந்தது. பணம் எடுப்பது ரத்து செய்யப்பட்டது.",
    },
    "no_voice_profile": {
        "english": "No voice profile is registered for this account, so withdrawals are not allowed. Please create a new account.",
        "hindi": "इस खाते के लिए कोई आवाज़ प्रोफ़ाइल पंजीकृत नहीं है, इसलिए निकासी की अनुमति नहीं है। कृपया एक नया खाता बनाएं।",
        "kannada": "ಈ ಖಾತೆಗೆ ಯಾವುದೇ ಧ್ವನಿ ಪ್ರೊಫೈಲ್ ನೋಂದಾಯಿಸಿಲ್ಲ, ಆದ್ದರಿಂದ ಹಿಂಪಡೆಯುವಿಕೆಗೆ ಅನುಮತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು ಹೊಸ ಖಾತೆ ರಚಿಸಿ.",
        "telugu": "ఈ ఖాతా కోసం వాయిస్ ప్రొఫైల్ నమోదు కాలేదు, కాబట్టి విత్‌డ్రా అనుమతించబడదు. దయచేసి కొత్త ఖాతాను సృష్టించండి.",
        "tamil": "இந்த கணக்கிற்கு குரல் சுயவிவரம் பதிவு செய்யப்படவில்லை, எனவே பணம் எடுப்பது அனுமதிக்கப்படாது. புதிய கணக்கை உருவாக்கவும்.",
    },
    "face_opening": {
        "english": "Opening the camera. Please face the screen and keep your head still. I will guide you.",
        "hindi": "कैमरा खोला जा रहा है। कृपया स्क्रीन की ओर देखें और अपना सिर स्थिर रखें। मैं आपका मार्गदर्शन करूंगा।",
        "kannada": "ಕ್ಯಾಮೆರಾ ತೆರೆಯಲಾಗುತ್ತಿದೆ. ದಯವಿಟ್ಟು ಪರದೆಯ ಕಡೆಗೆ ನೋಡಿ ಮತ್ತು ನಿಮ್ಮ ತಲೆಯನ್ನು ಸ್ಥಿರವಾಗಿಡಿ. ನಾನು ನಿಮಗೆ ಮಾರ್ಗದರ್ಶನ ನೀಡುತ್ತೇನೆ.",
        "telugu": "కెమెరా తెరవబడుతోంది. దయచేసి స్క్రీన్ వైపు చూడండి మరియు మీ తలను స్థిరంగా ఉంచండి. నేను మీకు మార్గనిర్దేశం చేస్తాను.",
        "tamil": "கேமரா திறக்கப்படுகிறது. தயவுசெய்து திரையை நோக்கிப் பாருங்கள், உங்கள் தலையை நிலையாக வையுங்கள். நான் உங்களுக்கு வழிகாட்டுவேன்.",
    },
    "camera_unavailable": {
        "english": "The camera is unavailable. Please close any other app that might be using it, such as Zoom, Teams, or another browser tab, and try again.",
        "hindi": "कैमरा उपलब्ध नहीं है। कृपया ज़ूम, टीम्स, या किसी अन्य ब्राउज़र टैब जैसे किसी अन्य ऐप को बंद करें जो इसका उपयोग कर रहा हो, और फिर से प्रयास करें।",
        "kannada": "ಕ್ಯಾಮೆರಾ ಲಭ್ಯವಿಲ್ಲ. ಇದನ್ನು ಬಳಸುತ್ತಿರುವ ಝೂಮ್, ಟೀಮ್ಸ್ ಅಥವಾ ಇತರ ಬ್ರೌಸರ್ ಟ್ಯಾಬ್‌ನಂತಹ ಯಾವುದೇ ಇತರ ಅಪ್ಲಿಕೇಶನ್ ಅನ್ನು ಮುಚ್ಚಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "telugu": "కెమెరా అందుబాటులో లేదు. దీన్ని ఉపయోగిస్తున్న జూమ్, టీమ్స్ లేదా మరొక బ్రౌజర్ ట్యాబ్ వంటి ఇతర యాప్‌ను మూసివేసి, మళ్లీ ప్రయత్నించండి.",
        "tamil": "கேமரா கிடைக்கவில்லை. இதைப் பயன்படுத்தும் Zoom, Teams அல்லது வேறு உலாவி தாவல் போன்ற பிற பயன்பாட்டை மூடிவிட்டு மீண்டும் முயற்சிக்கவும்.",
    },
    "move_closer": {
        "english": "Please move a little closer to the camera.",
        "hindi": "कृपया कैमरे के थोड़ा और करीब आएं।",
        "kannada": "ದಯವಿಟ್ಟು ಕ್ಯಾಮೆರಾಕ್ಕೆ ಸ್ವಲ್ಪ ಹತ್ತಿರ ಬನ್ನಿ.",
        "telugu": "దయచేసి కెమెరాకు కొంచెం దగ్గరగా రండి.",
        "tamil": "தயவுசெய்து கேமராவிற்கு கொஞ்சம் நெருக்கமாக வாருங்கள்.",
    },
    "move_back": {
        "english": "Please move a little back from the camera.",
        "hindi": "कृपया कैमरे से थोड़ा पीछे हटें।",
        "kannada": "ದಯವಿಟ್ಟು ಕ್ಯಾಮೆರಾದಿಂದ ಸ್ವಲ್ಪ ಹಿಂದೆ ಸರಿಯಿರಿ.",
        "telugu": "దయచేసి కెమెరా నుండి కొంచెం వెనక్కి వెళ్లండి.",
        "tamil": "தயவுசெய்து கேமராவிலிருந்து கொஞ்சம் பின்வாங்குங்கள்.",
    },
    "move_right": {
        "english": "Please move a little to your right.",
        "hindi": "कृपया अपनी दाईं ओर थोड़ा खिसकें।",
        "kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಬಲಕ್ಕೆ ಸ್ವಲ್ಪ ಸರಿಯಿರಿ.",
        "telugu": "దయచేసి మీ కుడి వైపుకు కొంచెం జరగండి.",
        "tamil": "தயவுசெய்து உங்கள் வலது புறம் கொஞ்சம் நகருங்கள்.",
    },
    "move_left": {
        "english": "Please move a little to your left.",
        "hindi": "कृपया अपनी बाईं ओर थोड़ा खिसकें।",
        "kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಎಡಕ್ಕೆ ಸ್ವಲ್ಪ ಸರಿಯಿರಿ.",
        "telugu": "దయచేసి మీ ఎడమ వైపుకు కొంచెం జరగండి.",
        "tamil": "தயவுசெய்து உங்கள் இடது புறம் கொஞ்சம் நகருங்கள்.",
    },
    "lower_head": {
        "english": "Please lower your head a little.",
        "hindi": "कृपया अपना सिर थोड़ा नीचे करें।",
        "kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ತಲೆಯನ್ನು ಸ್ವಲ್ಪ ಕೆಳಗೆ ಮಾಡಿ.",
        "telugu": "దయచేసి మీ తలను కొంచెం కిందికి దించండి.",
        "tamil": "தயவுசெய்து உங்கள் தலையை கொஞ்சம் கீழே தாழ்த்துங்கள்.",
    },
    "raise_head": {
        "english": "Please raise your head a little.",
        "hindi": "कृपया अपना सिर थोड़ा ऊपर उठाएं।",
        "kannada": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ತಲೆಯನ್ನು ಸ್ವಲ್ಪ ಮೇಲಕ್ಕೆ ಎತ್ತಿ.",
        "telugu": "దయచేసి మీ తలను కొంచెం పైకి ఎత్తండి.",
        "tamil": "தயவுசெய்து உங்கள் தலையை கொஞ்சம் உயர்த்துங்கள்.",
    },
    "face_side_profile": {
        "english": "I can see the side of your face. Please turn your face toward the camera.",
        "hindi": "मुझे आपके चेहरे का साइड दिख रहा है। कृपया अपना चेहरा कैमरे की ओर करें।",
        "kannada": "ನಿಮ್ಮ ಮುಖದ ಬದಿ ನನಗೆ ಕಾಣುತ್ತಿದೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಮುಖವನ್ನು ಕ್ಯಾಮೆರಾ ಕಡೆಗೆ ತಿರುಗಿಸಿ.",
        "telugu": "మీ ముఖం పక్క వైపు కనిపిస్తోంది. దయచేసి మీ ముఖాన్ని కెమెరా వైపు తిప్పండి.",
        "tamil": "உங்கள் முகத்தின் பக்கவாட்டைப் பார்க்கிறேன். தயவுசெய்து உங்கள் முகத்தை கேமரா பக்கம் திருப்புங்கள்.",
    },
    "face_not_found": {
        "english": "I cannot see your face. Please look toward the screen and move slowly until you hear a beep.",
        "hindi": "मुझे आपका चेहरा नहीं दिख रहा। कृपया स्क्रीन की ओर देखें और तब तक धीरे-धीरे हिलें जब तक आपको बीप सुनाई न दे।",
        "kannada": "ನನಗೆ ನಿಮ್ಮ ಮುಖ ಕಾಣುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಪರದೆಯ ಕಡೆಗೆ ನೋಡಿ ಮತ್ತು ಬೀಪ್ ಕೇಳುವವರೆಗೆ ನಿಧಾನವಾಗಿ ಚಲಿಸಿ.",
        "telugu": "నాకు మీ ముఖం కనిపించడం లేదు. దయచేసి స్క్రీన్ వైపు చూడండి మరియు బీప్ వినిపించే వరకు నెమ్మదిగా కదలండి.",
        "tamil": "எனக்கு உங்கள் முகம் தெரியவில்லை. தயவுசெய்து திரையை நோக்கிப் பார்த்து, பீப் கேட்கும் வரை மெதுவாக நகருங்கள்.",
    },
    "face_captured": {
        "english": "Face captured clearly.",
        "hindi": "चेहरा स्पष्ट रूप से कैद कर लिया गया।",
        "kannada": "ಮುಖವನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಸೆರೆಹಿಡಿಯಲಾಗಿದೆ.",
        "telugu": "ముఖం స్పష్టంగా క్యాప్చర్ చేయబడింది.",
        "tamil": "முகம் தெளிவாக படமெடுக்கப்பட்டது.",
    },
    "could_not_see_face": {
        "english": "I could not see your face. Please check the lighting and try again.",
        "hindi": "मुझे आपका चेहरा नहीं दिख सका। कृपया रोशनी जांचें और फिर से प्रयास करें।",
        "kannada": "ನನಗೆ ನಿಮ್ಮ ಮುಖ ಕಾಣಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಬೆಳಕನ್ನು ಪರಿಶೀಲಿಸಿ ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "telugu": "నాకు మీ ముఖం కనిపించలేదు. దయచేసి లైటింగ్‌ను తనిఖీ చేసి, మళ్లీ ప్రయత్నించండి.",
        "tamil": "எனக்கு உங்கள் முகம் தெரியவில்லை. தயவுசெய்து வெளிச்சத்தை சரிபார்த்து மீண்டும் முயற்சிக்கவும்.",
    },
    "camera_lost_feed": {
        "english": "I lost the camera feed. Please try again.",
        "hindi": "कैमरा फीड खो गया। कृपया फिर से प्रयास करें।",
        "kannada": "ಕ್ಯಾಮೆರಾ ಫೀಡ್ ಕಳೆದುಹೋಗಿದೆ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "telugu": "కెమెరా ఫీడ్ పోయింది. దయచేసి మళ్లీ ప్రయత్నించండి.",
        "tamil": "கேமரா ஃபீட் இழக்கப்பட்டது. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
    },
    "let_try_camera_again": {
        "english": "Let's try the camera once more.",
        "hindi": "आइए कैमरा एक बार फिर आज़माते हैं।",
        "kannada": "ಕ್ಯಾಮೆರಾವನ್ನು ಮತ್ತೊಮ್ಮೆ ಪ್ರಯತ್ನಿಸೋಣ.",
        "telugu": "కెమెరాను మరోసారి ప్రయత్నిద్దాం.",
        "tamil": "கேமராவை மீண்டும் ஒருமுறை முயற்சிப்போம்.",
    },
    "face_verification_passed": {
        "english": "Face verification passed.",
        "hindi": "चेहरा सत्यापन सफल रहा।",
        "kannada": "ಮುಖ ಪರಿಶೀಲನೆ ಯಶಸ್ವಿಯಾಗಿದೆ.",
        "telugu": "ముఖ ధృవీకరణ విజయవంతమైంది.",
        "tamil": "முக சரிபார்ப்பு வெற்றி பெற்றது.",
    },
    "face_verification_failed": {
        "english": "Face verification failed. This does not match the registered face.",
        "hindi": "चेहरा सत्यापन विफल रहा। यह पंजीकृत चेहरे से मेल नहीं खाता।",
        "kannada": "ಮುಖ ಪರಿಶೀಲನೆ ವಿಫಲವಾಗಿದೆ. ಇದು ನೋಂದಾಯಿತ ಮುಖಕ್ಕೆ ಹೊಂದಿಕೆಯಾಗುವುದಿಲ್ಲ.",
        "telugu": "ముఖ ధృవీకరణ విఫలమైంది. ఇది నమోదైన ముఖంతో సరిపోలడం లేదు.",
        "tamil": "முக சரிபார்ப்பு தோல்வியடைந்தது. இது பதிவு செய்யப்பட்ட முகத்துடன் பொருந்தவில்லை.",
    },
    "face_verification_failed_access": {
        "english": "Face verification failed. Access denied.",
        "hindi": "चेहरा सत्यापन विफल रहा। प्रवेश अस्वीकृत।",
        "kannada": "ಮುಖ ಪರಿಶೀಲನೆ ವಿಫಲವಾಗಿದೆ. ಪ್ರವೇಶ ನಿರಾಕರಿಸಲಾಗಿದೆ.",
        "telugu": "ముఖ ధృవీకరణ విఫలమైంది. ప్రవేశం నిరాకరించబడింది.",
        "tamil": "முக சரிபார்ப்பு தோல்வியடைந்தது. அணுகல் மறுக்கப்பட்டது.",
    },
    "face_verification_failed_withdraw": {
        "english": "Face verification failed. Withdrawal cancelled.",
        "hindi": "चेहरा सत्यापन विफल रहा। निकासी रद्द कर दी गई।",
        "kannada": "ಮುಖ ಪರಿಶೀಲನೆ ವಿಫಲವಾಗಿದೆ. ಹಿಂಪಡೆಯುವಿಕೆ ರದ್ದುಗೊಳಿಸಲಾಗಿದೆ.",
        "telugu": "ముఖ ధృవీకరణ విఫలమైంది. విత్‌డ్రా రద్దు చేయబడింది.",
        "tamil": "முக சரிபார்ப்பு தோல்வியடைந்தது. பணம் எடுப்பது ரத்து செய்யப்பட்டது.",
    },
    "no_face_registered": {
        "english": "No registered face was found for this account. Access denied.",
        "hindi": "इस खाते के लिए कोई पंजीकृत चेहरा नहीं मिला। प्रवेश अस्वीकृत।",
        "kannada": "ಈ ಖಾತೆಗೆ ಯಾವುದೇ ನೋಂದಾಯಿತ ಮುಖ ಕಂಡುಬಂದಿಲ್ಲ. ಪ್ರವೇಶ ನಿರಾಕರಿಸಲಾಗಿದೆ.",
        "telugu": "ఈ ఖాతా కోసం నమోదైన ముఖం కనుగొనబడలేదు. ప్రవేశం నిరాకరించబడింది.",
        "tamil": "இந்த கணக்கிற்கு பதிவு செய்யப்பட்ட முகம் எதுவும் இல்லை. அணுகல் மறுக்கப்பட்டது.",
    },
    "account_created": {
        "english": "Account created successfully. Your account number is {account_id}. Please remember this number.",
        "hindi": "खाता सफलतापूर्वक बनाया गया। आपका खाता नंबर {account_id} है। कृपया यह नंबर याद रखें।",
        "kannada": "ಖಾತೆಯನ್ನು ಯಶಸ್ವಿಯಾಗಿ ರಚಿಸಲಾಗಿದೆ. ನಿಮ್ಮ ಖಾತೆ ಸಂಖ್ಯೆ {account_id}. ದಯವಿಟ್ಟು ಈ ಸಂಖ್ಯೆಯನ್ನು ನೆನಪಿಡಿ.",
        "telugu": "ఖాతా విజయవంతంగా సృష్టించబడింది. మీ ఖాతా నంబర్ {account_id}. దయచేసి ఈ నంబర్‌ను గుర్తుంచుకోండి.",
        "tamil": "கணக்கு வெற்றிகரமாக உருவாக்கப்பட்டது. உங்கள் கணக்கு எண் {account_id}. இந்த எண்ணை நினைவில் வையுங்கள்.",
    },
    "creating_account": {
        "english": "Creating your account, {name}.",
        "hindi": "आपका खाता बनाया जा रहा है, {name}।",
        "kannada": "ನಿಮ್ಮ ಖಾತೆಯನ್ನು ರಚಿಸಲಾಗುತ್ತಿದೆ, {name}.",
        "telugu": "మీ ఖాతా సృష్టించబడుతోంది, {name}.",
        "tamil": "உங்கள் கணக்கு உருவாக்கப்படுகிறது, {name}.",
    },
    "voice_registration_prompt": {
        "english": "Now let's register your voice for withdrawal security. After the beep, say clearly: my voice is my password.",
        "hindi": "अब हम निकासी सुरक्षा के लिए आपकी आवाज़ पंजीकृत करेंगे। बीप के बाद, स्पष्ट रूप से कहें: मेरी आवाज़ ही मेरा पासवर्ड है।",
        "kannada": "ಈಗ ಹಿಂಪಡೆಯುವಿಕೆ ಭದ್ರತೆಗಾಗಿ ನಿಮ್ಮ ಧ್ವನಿಯನ್ನು ನೋಂದಾಯಿಸೋಣ. ಬೀಪ್ ನಂತರ, ಸ್ಪಷ್ಟವಾಗಿ ಹೇಳಿ: ನನ್ನ ಧ್ವನಿಯೇ ನನ್ನ ಪಾಸ್‌ವರ್ಡ್.",
        "telugu": "ఇప్పుడు విత్‌డ్రా భద్రత కోసం మీ వాయిస్‌ను నమోదు చేద్దాం. బీప్ తర్వాత, స్పష్టంగా చెప్పండి: నా వాయిస్ నా పాస్‌వర్డ్.",
        "tamil": "இப்போது பணம் எடுப்பு பாதுகாப்புக்காக உங்கள் குரலைப் பதிவு செய்வோம். பீப் ஒலிக்குப் பிறகு, தெளிவாகச் சொல்லுங்கள்: என் குரலே என் கடவுச்சொல்.",
    },
    "voice_registered": {
        "english": "Voice registered successfully.",
        "hindi": "आवाज़ सफलतापूर्वक पंजीकृत हो गई।",
        "kannada": "ಧ್ವನಿಯನ್ನು ಯಶಸ್ವಿಯಾಗಿ ನೋಂದಾಯಿಸಲಾಗಿದೆ.",
        "telugu": "వాయిస్ విజయవంతంగా నమోదు చేయబడింది.",
        "tamil": "குரல் வெற்றிகரமாக பதிவு செய்யப்பட்டது.",
    },
    "voice_not_heard": {
        "english": "I did not hear you.",
        "hindi": "मुझे आपकी आवाज़ सुनाई नहीं दी।",
        "kannada": "ನನಗೆ ನಿಮ್ಮ ಧ್ವನಿ ಕೇಳಿಸಲಿಲ್ಲ.",
        "telugu": "నాకు మీ వాయిస్ వినిపించలేదు.",
        "tamil": "எனக்கு உங்கள் குரல் கேட்கவில்லை.",
    },
    "could_not_register_face": {
        "english": "I could not register your face, so the account was not created. Please try again.",
        "hindi": "मैं आपका चेहरा पंजीकृत नहीं कर सका, इसलिए खाता नहीं बनाया गया। कृपया फिर से प्रयास करें।",
        "kannada": "ನಿಮ್ಮ ಮುಖವನ್ನು ನೋಂದಾಯಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ಖಾತೆ ರಚಿಸಲಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "telugu": "మీ ముఖాన్ని నమోదు చేయలేకపోయాను, కాబట్టి ఖాతా సృష్టించబడలేదు. దయచేసి మళ్లీ ప్రయత్నించండి.",
        "tamil": "உங்கள் முகத்தை பதிவு செய்ய முடியவில்லை, எனவே கணக்கு உருவாக்கப்படவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
    },
    "could_not_register_voice": {
        "english": "I could not register your voice, so the account was not created. Please try again.",
        "hindi": "मैं आपकी आवाज़ पंजीकृत नहीं कर सका, इसलिए खाता नहीं बनाया गया। कृपया फिर से प्रयास करें।",
        "kannada": "ನಿಮ್ಮ ಧ್ವನಿಯನ್ನು ನೋಂದಾಯಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ಖಾತೆ ರಚಿಸಲಾಗಿಲ್ಲ. ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "telugu": "మీ వాయిస్‌ను నమోదు చేయలేకపోయాను, కాబట్టి ఖాతా సృష్టించబడలేదు. దయచేసి మళ్లీ ప్రయత్నించండి.",
        "tamil": "உங்கள் குரலைப் பதிவு செய்ய முடியவில்லை, எனவே கணக்கு உருவாக்கப்படவில்லை. தயவுசெய்து மீண்டும் முயற்சிக்கவும்.",
    },
    "speech_service_unreachable": {
        "english": "Speech recognition service is unreachable. Please check your internet connection.",
        "hindi": "स्पीच पहचान सेवा उपलब्ध नहीं है। कृपया अपना इंटरनेट कनेक्शन जांचें।",
        "kannada": "ಸ್ಪೀಚ್ ರೆಕಗ್ನಿಷನ್ ಸೇವೆ ತಲುಪಲಾಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಇಂಟರ್ನೆಟ್ ಸಂಪರ್ಕವನ್ನು ಪರಿಶೀಲಿಸಿ.",
        "telugu": "స్పీచ్ రికగ్నిషన్ సేవను చేరుకోలేకపోయాము. దయచేసి మీ ఇంటర్నెట్ కనెక్షన్‌ను తనిఖీ చేయండి.",
        "tamil": "பேச்சு அடையாள சேவையை அணுக முடியவில்லை. உங்கள் இணைய இணைப்பைச் சரிபார்க்கவும்.",
    },
}

KIND_LABELS = {
    "DEPOSIT": {"english": "deposit", "hindi": "जमा", "kannada": "ಠೇವಣಿ", "telugu": "డిపాజిట్", "tamil": "டெபாசிட்"},
    "WITHDRAW": {"english": "withdrawal", "hindi": "निकासी", "kannada": "ಹಿಂಪಡೆಯುವಿಕೆ", "telugu": "విత్‌డ్రా", "tamil": "பணம் எடுப்பு"},
}

STATIC_TRANSLATIONS = {
    "You have been logged out safely.": {
        "hindi": MESSAGES["logout_manual"]["hindi"],
        "kannada": MESSAGES["logout_manual"]["kannada"],
        "telugu": MESSAGES["logout_manual"]["telugu"],
        "tamil": MESSAGES["logout_manual"]["tamil"],
    },
    "You can also press Create Account or Login on the screen.": {
        "hindi": MESSAGES["mode_not_understood"]["hindi"],
        "kannada": MESSAGES["mode_not_understood"]["kannada"],
        "telugu": MESSAGES["mode_not_understood"]["telugu"],
        "tamil": MESSAGES["mode_not_understood"]["tamil"],
    },
    "No response was heard, so for your security you have been logged out.": {
        "hindi": MESSAGES["logout_no_response"]["hindi"],
        "kannada": MESSAGES["logout_no_response"]["kannada"],
        "telugu": MESSAGES["logout_no_response"]["telugu"],
        "tamil": MESSAGES["logout_no_response"]["tamil"],
    },
}


# Strips ONLY punctuation. (A regex like [^\w\s] would also delete Indic vowel
# signs such as the ि in हिंदी, corrupting Hindi/Kannada/Telugu/Tamil text.)
_PUNCT_TABLE = str.maketrans({c: ' ' for c in '.,!?;:"\'()[]{}<>-_/\\|।॥…'})


def _strip_punct(text):
    return (text or "").translate(_PUNCT_TABLE)


# ======================= FUZZY MATCHING =======================

def _is_subsequence(short_word, candidate):
    it = iter(candidate)
    return all(ch in it for ch in short_word)


def _normalize_stretched(word):
    return re.sub(r'(.)\1+', r'\1', word)


def fuzzy_match(word, keyword_set, cutoff=FUZZY_CUTOFF):
    word = word.strip().lower()
    if not word:
        return False
    if word in keyword_set:
        return True
    norm_word = _normalize_stretched(word)
    for kw in keyword_set:
        if norm_word == _normalize_stretched(kw.lower()):
            return True
    if get_close_matches(word, keyword_set, n=1, cutoff=cutoff):
        return True
    if 3 <= len(word) <= 5:
        for kw in keyword_set:
            kw_l = kw.lower()
            if (len(kw_l) > len(word) and ' ' not in kw_l
                    and kw_l[0] == word[0] and _is_subsequence(word, kw_l)):
                return True
    return False


def _best_ratio(word, keyword_set):
    best = 0.0
    norm_word = _normalize_stretched(word)
    for kw in keyword_set:
        kw_l = kw.lower()
        r = SequenceMatcher(None, word, kw_l).ratio()
        r_norm = SequenceMatcher(None, norm_word, _normalize_stretched(kw_l)).ratio()
        best = max(best, r, r_norm)
    return best


def fuzzy_yes_no(word, yes_set, no_set, cutoff=FUZZY_CUTOFF, loose_threshold=0.55):
    word = word.strip().lower()
    if not word:
        return None
    if word in yes_set:
        return 'yes'
    if word in no_set:
        return 'no'
    yes_fuzzy = fuzzy_match(word, yes_set, cutoff=cutoff)
    no_fuzzy = fuzzy_match(word, no_set, cutoff=cutoff)
    if yes_fuzzy and not no_fuzzy:
        return 'yes'
    if no_fuzzy and not yes_fuzzy:
        return 'no'
    yes_score = _best_ratio(word, yes_set)
    no_score = _best_ratio(word, no_set)
    if yes_score >= loose_threshold and yes_score > no_score:
        return 'yes'
    if no_score >= loose_threshold and no_score > yes_score:
        return 'no'
    return None


def fuzzy_contains_phrase(heard_text, phrase_list, cutoff=FUZZY_CUTOFF):
    heard = heard_text.strip().lower()
    if not heard:
        return False
    for phrase in phrase_list:
        phrase_l = phrase.lower()
        if phrase_l in heard:
            return True
        if ' ' in phrase_l:
            continue
        for w in heard.split():
            if fuzzy_match(w, {phrase_l}, cutoff=cutoff):
                return True
    return False


# ======================= LIVE TRANSLATION (fallback only) =======================
_translate_cache = {}
_translate_cache_lock = threading.Lock()
_TRANSLATE_COOLDOWN_SECONDS = 90
_translate_cooldown_until = 0.0
_translate_fail_streak = 0


def translate_text(text, target_lang_code, lang_key=None):
    """Last-resort live translation for text that has no MESSAGES entry."""
    global _translate_cooldown_until, _translate_fail_streak
    if not text or target_lang_code == "en":
        return text
    if lang_key:
        static = STATIC_TRANSLATIONS.get(text, {}).get(lang_key)
        if static:
            return static
    cache_key = (target_lang_code, text)
    with _translate_cache_lock:
        cached = _translate_cache.get(cache_key)
    if cached is not None:
        return cached
    if not _TRANSLATE_AVAILABLE or time.time() < _translate_cooldown_until:
        return text
    translated = None
    for attempt_fn in (
        lambda: GoogleTranslator(source='en', target=target_lang_code).translate(text),
        lambda: MyMemoryTranslator(source='en-GB', target=f"{target_lang_code}-IN").translate(text),
    ):
        try:
            result = attempt_fn()
            if result:
                translated = result
                break
        except Exception:
            continue
    if translated:
        _translate_fail_streak = 0
        with _translate_cache_lock:
            _translate_cache[cache_key] = translated
        return translated
    _translate_fail_streak += 1
    if _translate_fail_streak >= 2:
        _translate_cooldown_until = time.time() + _TRANSLATE_COOLDOWN_SECONDS
    return text


# ======================= SPEECH OUTPUT =======================

def _tts_cache_path(text, tts_lang):
    key = hashlib.md5(f"{tts_lang}:{text}".encode("utf-8")).hexdigest()
    return os.path.join(TTS_CACHE_DIR, f"{tts_lang}_{key}.mp3")


def _play_mp3(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.05)
    pygame.mixer.music.unload()


def _speak_gtts(text, tts_lang, retries=2):
    cache_path = _tts_cache_path(text, tts_lang)
    if os.path.exists(cache_path):
        try:
            _play_mp3(cache_path)
            return True
        except Exception:
            try:
                os.remove(cache_path)
            except Exception:
                pass
    last_error = None
    for attempt in range(retries + 1):
        try:
            tts = gTTS(text=text, lang=tts_lang)
            tts.save(cache_path)
            _play_mp3(cache_path)
            return True
        except Exception as e:
            last_error = e
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(0.4 * (attempt + 1))
    print(f"[gTTS Error]: {last_error} - falling back to offline voice.")
    return False


def _speak_pyttsx3(text):
    com_ready = False
    if pythoncom is not None:
        try:
            pythoncom.CoInitialize()
            com_ready = True
        except Exception:
            pass
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 145)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
    except Exception as e:
        print(f"[Speech Output Error]: {e}")
    finally:
        if com_ready:
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass


def speak_localized(text):
    print(f"\n[VoiceBank ({_lang_conf()['label']})]: {text}")
    with speak_lock:
        spoken = False
        if _GTTS_AVAILABLE:
            spoken = _speak_gtts(text, _lang_conf()["tts"])
        if not spoken:
            _speak_pyttsx3(text)


def say(key, **kwargs):
    template_set = MESSAGES.get(key)
    if template_set is None:
        return
    lang_key = get_language()
    text = template_set.get(lang_key) or template_set.get("english")
    if not text:
        return
    try:
        text = text.format(**kwargs)
    except Exception:
        pass
    speak_localized(text)


def speak(text):
    lang_key = get_language()
    conf = _lang_conf()
    spoken_text = text
    if conf["tts"] != "en":
        spoken_text = translate_text(text, conf["tts"], lang_key=lang_key)
    speak_localized(spoken_text)


# ======================= SMART LISTENING =======================

cancel_listen_event = threading.Event()


def request_cancel_listen():
    cancel_listen_event.set()


_device_logged = False


def _log_input_device_once():
    """Prints which microphone Windows is actually giving us, once."""
    global _device_logged
    if _device_logged:
        return
    _device_logged = True
    try:
        info = sd.query_devices(kind='input')
        print(f"[Mic] using input device: {info['name']}")
    except Exception as e:
        print(f"[Mic] could not query input device: {e}")


def record_until_silence(max_wait=10, max_total=DEFAULT_MAX_TOTAL, silence_end=1.5):
    """Records one utterance.

    Start: noise is measured on a rolling window of non-speech blocks; speech
    starts after SPEECH_START_BLOCKS loud blocks in a row.
    End: the 'still speaking' level is RELATIVE TO THE SPEAKER'S OWN LEVEL
    (median of voiced blocks), so background noise of any kind can no longer
    keep the recording open until the time limit.
    """
    cancel_listen_event.clear()
    _log_input_device_once()
    block_size = int(SAMPLE_RATE * BLOCK_SECONDS)
    blocks = []
    recent = deque(maxlen=NOISE_WINDOW_BLOCKS)
    voiced_levels = deque(maxlen=40)
    noise = 0.0
    threshold = float(VAD_MIN_THRESHOLD)
    end_level = threshold * 0.6
    speech_started = False
    consecutive = 0
    silent_run = 0
    first_voiced = None
    last_voiced = None
    loudest = 0.0
    start = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', blocksize=block_size) as stream:
        while True:
            if cancel_listen_event.is_set():
                return None
            data, _ = stream.read(block_size)
            block = data[:, 0].copy()
            idx = len(blocks)
            blocks.append(block)
            rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
            elapsed = time.time() - start

            if idx < CALIBRATION_SKIP_BLOCKS:
                continue
            loudest = max(loudest, rms)

            if not speech_started:
                if len(recent) >= 5:
                    noise = float(np.percentile(recent, NOISE_PERCENTILE))
                    threshold = min(max(noise * VAD_NOISE_MULTIPLIER, VAD_MIN_THRESHOLD),
                                    VAD_MAX_THRESHOLD)
                if rms > threshold:
                    consecutive += 1
                    voiced_levels.append(rms)
                    if consecutive >= SPEECH_START_BLOCKS or rms > threshold * 4:
                        speech_started = True
                        first_voiced = idx - consecutive + 1
                        last_voiced = idx
                        silent_run = 0
                        print(f"[Mic] speech detected (noise {noise:.0f}, threshold {threshold:.0f})")
                else:
                    consecutive = 0
                    voiced_levels.clear()
                    recent.append(rms)
                if not speech_started and elapsed >= max_wait:
                    break
            else:
                if rms > threshold:
                    voiced_levels.append(rms)
                speech_ref = float(np.median(voiced_levels)) if voiced_levels else threshold
                end_level = max(threshold * 0.6, speech_ref * END_LEVEL_FRACTION)
                if rms > end_level:
                    last_voiced = idx
                    silent_run = 0
                else:
                    silent_run += 1
                if silent_run * BLOCK_SECONDS >= silence_end:
                    break

            if elapsed >= max_total:
                break

    if not speech_started:
        print(f"[Mic] no speech detected - loudest sound heard was {loudest:.0f} "
              f"(needed more than {threshold:.0f}).")
        return None
    if (last_voiced - first_voiced) < MIN_SPEECH_BLOCKS:
        print("[Mic] sound was too short to be speech.")
        return None

    start_i = max(0, first_voiced - PRE_ROLL_BLOCKS)
    end_i = min(len(blocks), last_voiced + POST_ROLL_BLOCKS + 1)
    audio = np.concatenate(blocks[start_i:end_i])
    print(f"[Mic] recorded {len(audio) / SAMPLE_RATE:.1f}s, peak level {int(np.max(np.abs(audio)))}, "
          f"end-level {end_level:.0f}")
    return audio


def save_wav(path, audio):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def _denoise_for_recognition(audio):
    """Noise cleanup for speech recognition ONLY (never for the biometric sample)."""
    if not _NOISEREDUCE_AVAILABLE or audio is None or len(audio) == 0:
        return audio
    try:
        float_audio = audio.astype(np.float32) / 32768.0
        cleaned = nr.reduce_noise(y=float_audio, sr=SAMPLE_RATE, stationary=False)
        cleaned = np.clip(cleaned * 32768.0, -32768, 32767)
        return cleaned.astype(np.int16)
    except Exception as e:
        print(f"[Noise Reduction Error]: {e} - using the original recording.")
        return audio


# ---------------------- Whisper input preparation ----------------------

def _prepare_for_whisper(audio_int16):
    """int16 recording -> clean, loud, length-capped float32 array for Whisper."""
    x = audio_int16.astype(np.float32) / 32768.0
    if x.size == 0:
        return x
    x = x[: int(SAMPLE_RATE * MAX_WHISPER_SECONDS)]      # hard cap = bounded decode time
    x = x - float(np.mean(x))

    if _SCIPY_AVAILABLE:
        try:
            sos = butter(2, WHISPER_HIGHPASS_HZ / (SAMPLE_RATE / 2.0), btype='highpass', output='sos')
            x = sosfilt(sos, x).astype(np.float32)
        except Exception:
            pass

    peak = float(np.max(np.abs(x)))
    if peak > 1e-5:
        gain = min(WHISPER_TARGET_PEAK / peak, WHISPER_MAX_GAIN)
        x = np.clip(x * gain, -1.0, 1.0).astype(np.float32)

    pad = np.zeros(int(SAMPLE_RATE * WHISPER_EDGE_PAD_SECONDS), dtype=np.float32)
    return np.concatenate([pad, x, pad])


_WHISPER_HALLUCINATIONS = {
    "", "you", "thank you", "thanks", "thanks for watching", "thank you for watching",
    "bye", "the", "uh", "um", "hmm", "mm", "mhm", ".", "..",
}


def _collapse_repeats(text):
    """Whisper loops on noise ('log in, log in, log in...' / 'दादादादा...').
    Collapse those loops to one copy. Spoken digit strings (PINs like
    'one one one one' or '1 1 1 1') are left untouched."""
    cleaned = " ".join(_strip_punct(text).split())
    tokens = cleaned.split()
    if not tokens:
        return text
    if all(t in _DIGIT_WORDS or t.isdigit() for t in tokens):
        return text
    out = re.sub(r'(\S+(?:\s+\S+){0,3}?)(?:\s+\1){2,}', r'\1', cleaned)   # repeated words/phrases
    out = re.sub(r'(\D{1,8}?)\1{4,}', r'\1', out)                          # repeated characters
    return out if out != cleaned else text


def _clean_transcript(text):
    text = (text or "").strip().lower()
    text = _collapse_repeats(text)
    core = " ".join(_strip_punct(text).split())
    if core in _WHISPER_HALLUCINATIONS:
        return ""
    if not re.fullmatch(r'[\d\s]+', core):
        compact = core.replace(' ', '')
        if len(compact) >= 8 and len(set(compact)) <= 3:
            return ""
        if re.search(r'(.)\1{5,}', compact):
            return ""
    return text


def build_recognition_hint(kind):
    conf = _lang_conf()
    if kind == "digits":
        return ("Indian English speaker saying a number digit by digit: "
                "zero, one, two, three, four, five, six, seven, eight, nine.")
    if kind == "command":
        words = []
        for phrases in conf["commands"].values():
            words.extend(phrases)
        for phrases in LANGUAGES["english"]["commands"].values():
            words.extend(phrases)
        return "Indian speaker giving a banking command: " + ", ".join(dict.fromkeys(words)) + "."
    if kind == "mode":
        words = []
        for phrases in conf.get("mode", {}).values():
            words.extend(phrases)
        for phrases in LANGUAGES["english"]["mode"].values():
            words.extend(phrases)
        return "Indian speaker saying: " + ", ".join(dict.fromkeys(words)) + "."
    if kind == "yesno":
        words = sorted(conf["yes"] | conf["no"] | LANGUAGES["english"]["yes"] | LANGUAGES["english"]["no"])
        return "Indian speaker answering yes or no: " + ", ".join(words) + "."
    if kind == "language":
        return "Indian speaker choosing a language: English, Hindi, Kannada, Telugu, Tamil."
    return None


def transcribe(source, return_alternatives=False, extra_langs=None, hint=None):
    """Runs Whisper (offline). Greedy decoding + no timestamps = several times
    faster than beam search, and the audio is already length-capped."""
    langs_to_try = [_lang_conf()["stt"].split('-')[0]]
    for lang_code in (extra_langs or []):
        base = lang_code.split('-')[0]
        if base not in langs_to_try:
            langs_to_try.append(base)

    empty = ([], "") if return_alternatives else ""

    if not _WHISPER_AVAILABLE or _whisper_model is None:
        print("[Transcribe Error]: Whisper model is not available.")
        return empty

    initial_prompt = build_recognition_hint(hint) if hint else None

    alternatives = []
    for whisper_lang in langs_to_try:
        try:
            t0 = time.time()
            result = _whisper_model.transcribe(
                source,
                language=whisper_lang,
                fp16=False,
                condition_on_previous_text=False,
                no_speech_threshold=None,
                logprob_threshold=None,
                temperature=0.0,
                without_timestamps=True,
                initial_prompt=initial_prompt,
            )
            print(f"[Whisper ({whisper_lang})] {time.time() - t0:.1f}s")
            text = _clean_transcript(result.get("text"))
        except Exception as e:
            print(f"[Transcribe Error ({whisper_lang})]: {e}")
            continue
        if text and text not in alternatives:
            alternatives.append(text)

    best = alternatives[0] if alternatives else ""
    if return_alternatives:
        return alternatives, best
    return best


def fallback_stt_langs():
    stt = _lang_conf()["stt"]
    return [] if stt == "en-IN" else ["en-IN"]


def listen(max_wait=10, max_total=DEFAULT_MAX_TOTAL, silence_end=1.5, save_voice_path=None,
           return_alternatives=False, extra_langs=None, hint=None):
    """Records and transcribes one utterance."""
    filename = save_voice_path if save_voice_path else None
    empty = ([], "") if return_alternatives else ""
    time.sleep(PRE_LISTEN_SETTLE_SECONDS)
    print("\n[VoiceBank is listening... speak after the beep]")
    winsound.Beep(1000, 250)
    try:
        audio = record_until_silence(max_wait=max_wait, max_total=max_total, silence_end=silence_end)
        if audio is None:
            if cancel_listen_event.is_set():
                return empty
            say("nothing_heard")
            return empty

        if filename:
            save_wav(filename, audio)

        prepared = _prepare_for_whisper(audio)

        if return_alternatives:
            alternatives, best = transcribe(prepared, return_alternatives=True,
                                            extra_langs=extra_langs, hint=hint)
            if not alternatives:
                denoised = _denoise_for_recognition(audio)
                if denoised is not audio:
                    alternatives, best = transcribe(_prepare_for_whisper(denoised),
                                                    return_alternatives=True,
                                                    extra_langs=extra_langs, hint=hint)
            if alternatives:
                print(f"[Recognized Speech Alternatives]: {alternatives}")
            else:
                say("not_understood")
            return alternatives, best

        text = transcribe(prepared, extra_langs=extra_langs, hint=hint)
        if not text:
            denoised = _denoise_for_recognition(audio)
            if denoised is not audio:
                text = transcribe(_prepare_for_whisper(denoised), extra_langs=extra_langs, hint=hint)
        if text:
            print(f"[Recognized Speech]: {text}")
        else:
            say("not_understood")
        return text
    except Exception as e:
        print(f"[Voice Capture Notice]: {e}")
        return empty


def capture_voice_sample(path, max_wait=10, max_total=8):
    """Records a voice sample for biometrics (raw signal, no processing)."""
    print("\n[VoiceBank is listening for your voice sample... speak after the beep]")
    time.sleep(PRE_LISTEN_SETTLE_SECONDS)
    winsound.Beep(1000, 250)
    try:
        audio = record_until_silence(max_wait=max_wait, max_total=max_total, silence_end=1.5)
        if audio is None:
            return False
        save_wav(path, audio)
        return True
    except Exception as e:
        print(f"[Voice Sample Error]: {e}")
        return False


# ======================= SPOKEN NUMBERS, AMOUNTS, COMMANDS =======================

# Common ways Indian-accented speech (and Whisper's guesses at it) comes out
# for the digits, plus Hindi number words.
_DIGIT_WORDS = {
    'zero': '0', 'oh': '0', 'o': '0', 'ziro': '0', 'jero': '0', 'shunya': '0', 'sifar': '0',
    'one': '1', 'won': '1', 'wan': '1', 'ek': '1',
    'two': '2', 'to': '2', 'too': '2', 'tu': '2', 'do': '2',
    'three': '3', 'tree': '3', 'thri': '3', 'free': '3', 'teen': '3',
    'four': '4', 'for': '4', 'fore': '4', 'faur': '4', 'char': '4', 'chaar': '4',
    'five': '5', 'fife': '5', 'fibe': '5', 'paanch': '5', 'panch': '5',
    'six': '6', 'sicks': '6', 'chhe': '6', 'chhah': '6', 'che': '6',
    'seven': '7', 'seben': '7', 'saat': '7', 'sat': '7',
    'eight': '8', 'ate': '8', 'ait': '8', 'eat': '8', 'aath': '8', 'ath': '8',
    'nine': '9', 'nain': '9', 'nau': '9', 'nao': '9',
}


def _ascii_digits(text):
    """Converts Devanagari / Kannada / Telugu / Tamil numerals to 0-9."""
    out = []
    for c in text:
        if c.isdigit():
            try:
                out.append(str(unicodedata.digit(c)))
                continue
            except (ValueError, TypeError):
                pass
        out.append(c)
    return "".join(out)


_MULTI_DIGIT_WORDS = {
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
    'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18', 'nineteen': '19',
}
_REPEAT_WORDS = {
    'double': 2, 'dabal': 2, 'dubble': 2, 'duble': 2,
    'triple': 3, 'tripple': 3, 'tripal': 3,
}


def extract_numbers(text):
    """Turns what was heard into a digit string. Understands digits, English
    words, Indian-accent variants, Hindi number words, Indic numerals,
    'double/triple', and ten..nineteen."""
    if not text:
        return ""
    text = _ascii_digits(text)
    single_words = set(_DIGIT_WORDS.keys())
    converted = ""
    repeat = 1
    for w in re.split(r'[\s,.\-]+', text.lower()):
        if not w:
            continue
        if w in _REPEAT_WORDS:
            repeat = _REPEAT_WORDS[w]
            continue
        piece = None
        if w in _DIGIT_WORDS:
            piece = _DIGIT_WORDS[w]
        elif w in _MULTI_DIGIT_WORDS:
            piece = _MULTI_DIGIT_WORDS[w]
        else:
            digits_in_word = "".join(c for c in w if c.isdigit())
            if digits_in_word:
                piece = digits_in_word
            else:
                match = get_close_matches(w, single_words, n=1, cutoff=0.75)
                if match:
                    piece = _DIGIT_WORDS[match[0]]
        if piece is not None:
            converted += piece * repeat
            repeat = 1
    return converted


_UNITS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
          'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
          'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
          'seventeen': 17, 'eighteen': 18, 'nineteen': 19}
_TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
         'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
_SCALES = {'thousand': 1_000, 'lakh': 100_000, 'lakhs': 100_000,
           'million': 1_000_000, 'crore': 10_000_000, 'crores': 10_000_000}


def parse_amount(text):
    if not text:
        return None
    t = _ascii_digits(text.lower()).replace(',', '')
    m = re.search(r'\d+(?:\.\d+)?', t)
    if m:
        value = float(m.group())
        rest = t[m.end():].split()
        if rest and rest[0] in _SCALES:
            value *= _SCALES[rest[0]]
        elif rest and rest[0] == 'hundred':
            value *= 100
        return value

    total, current, found = 0, 0, False
    for w in t.split():
        w = w.strip('.,')
        if w in _UNITS:
            current += _UNITS[w]
            found = True
        elif w in _TENS:
            current += _TENS[w]
            found = True
        elif w == 'hundred':
            current = max(current, 1) * 100
            found = True
        elif w in _SCALES:
            total += max(current, 1) * _SCALES[w]
            current = 0
            found = True
    total += current
    return float(total) if found else None


def _command_sets():
    """Commands for the active language, plus English (people mix languages)."""
    sets = [_lang_conf()["commands"]]
    if get_language() != "english":
        sets.append(LANGUAGES["english"]["commands"])
    return sets


def parse_command(text, cutoff=FUZZY_CUTOFF_STRICT):
    if not text:
        return 'none'
    for commands in _command_sets():
        for action, phrases in commands.items():
            if fuzzy_contains_phrase(text, phrases, cutoff=cutoff):
                return action
    return 'unknown'


def pick_command(alternatives, cutoff=FUZZY_CUTOFF_STRICT):
    for alt in alternatives:
        action = parse_command(alt, cutoff=cutoff)
        if action not in ('none', 'unknown'):
            return action, alt
    if alternatives:
        return parse_command(alternatives[0], cutoff=cutoff), alternatives[0]
    return 'none', ''


def ask_command(prompt_key, attempts=2):
    last_action, last_heard = 'none', ''
    for attempt in range(attempts):
        if attempt == 0:
            say(prompt_key)
        else:
            say("please_repeat_command")
        cutoff = max(0.62, FUZZY_CUTOFF_STRICT - 0.09 * attempt)
        alternatives, _top = listen(max_wait=12, silence_end=1.5, return_alternatives=True,
                                     extra_langs=fallback_stt_langs(), hint="command")
        if not alternatives:
            last_action, last_heard = 'none', ''
            continue
        action, matched = pick_command(alternatives, cutoff=cutoff)
        last_action, last_heard = action, matched
        if action not in ('none', 'unknown'):
            return action, matched
    return last_action, last_heard


def parse_mode(text, cutoff=FUZZY_CUTOFF_STRICT):
    if not text:
        return 'none'
    mode_sets = [_lang_conf().get("mode", {})]
    if get_language() != "english":
        mode_sets.append(LANGUAGES["english"]["mode"])
    for modes in mode_sets:
        for key, phrases in modes.items():
            if fuzzy_contains_phrase(text, phrases, cutoff=cutoff):
                return key
    return 'unknown'


def pick_mode(alternatives, cutoff=FUZZY_CUTOFF_STRICT):
    for alt in alternatives:
        mode = parse_mode(alt, cutoff=cutoff)
        if mode not in ('none', 'unknown'):
            return mode, alt
    if alternatives:
        return 'unknown', alternatives[0]
    return 'none', ''


def ask_mode_choice(attempts=3):
    for attempt in range(attempts):
        cutoff = max(0.58, FUZZY_CUTOFF_STRICT - 0.07 * attempt)
        say("ask_mode")
        alternatives, _top = listen(max_wait=10, silence_end=1.3, return_alternatives=True,
                                     extra_langs=fallback_stt_langs(), hint="mode")
        if not alternatives:
            continue
        mode, _matched = pick_mode(alternatives, cutoff=cutoff)
        if mode in ('create', 'login'):
            return mode
    return None


def _match_language_name(text):
    """Finds which language was named, however Whisper chose to write it:
    Latin ('telugu'), Devanagari ('तेलुगु'), or the native script. Spaces are
    ignored ('ते लुगू' still matches) and near-misses are accepted."""
    if not text:
        return None
    cleaned = " ".join(_strip_punct(text.lower()).split())
    compact = cleaned.replace(" ", "")
    if not compact:
        return None
    candidates = cleaned.split() + [compact]
    best_key, best_score = None, 0.0
    for key, conf in LANGUAGES.items():
        for name in conf.get("names", []):
            n = name.lower().replace(" ", "")
            if len(n) >= 3 and n in compact:
                return key
            for cand in candidates:
                score = SequenceMatcher(None, cand, n).ratio()
                if score > best_score:
                    best_score, best_key = score, key
    return best_key if best_score >= 0.70 else None


def ask_language_choice(attempts=3):
    for attempt in range(attempts):
        say("ask_language")
        # Two passes (English + Hindi) so the language name is caught
        # whichever script Whisper writes it in.
        alternatives, _top = listen(max_wait=10, silence_end=1.5, return_alternatives=True,
                                     extra_langs=["hi"], hint="language")
        if not alternatives:
            continue
        for alt in alternatives:
            key = _match_language_name(alt)
            if key:
                return key
        if attempt < attempts - 1:
            say("language_not_understood")
    return None


def ask_yes_no(message_key, attempts=3, **kwargs):
    """Asks a yes/no question and returns True/False. Used ONLY for
    confirming a money amount."""
    conf = _lang_conf()
    yes_words = conf["yes"] | LANGUAGES["english"]["yes"]
    no_words = conf["no"] | LANGUAGES["english"]["no"]

    for attempt in range(attempts):
        cutoff = max(0.52, FUZZY_CUTOFF - 0.06 * attempt)
        loose_threshold = max(0.40, 0.55 - 0.05 * attempt)

        say(message_key, **kwargs)
        alternatives, _top = listen(max_wait=8, max_total=8, silence_end=1.3,
                                     return_alternatives=True, extra_langs=fallback_stt_langs(),
                                     hint="yesno")
        if not alternatives:
            continue

        for alt in alternatives:
            words = _strip_punct(alt).split()
            if any(w in no_words for w in words) or alt.strip() in no_words:
                return False
            if any(w in yes_words for w in words) or alt.strip() in yes_words:
                return True

        for alt in alternatives:
            if fuzzy_contains_phrase(alt, no_words, cutoff=cutoff):
                return False
        for alt in alternatives:
            for w in alt.replace(',', ' ').split():
                if fuzzy_yes_no(w, yes_words, no_words, cutoff=cutoff, loose_threshold=loose_threshold) == 'no':
                    return False

        for alt in alternatives:
            if fuzzy_contains_phrase(alt, yes_words, cutoff=cutoff):
                return True
        for alt in alternatives:
            for w in alt.replace(',', ' ').split():
                if fuzzy_yes_no(w, yes_words, no_words, cutoff=cutoff, loose_threshold=loose_threshold) == 'yes':
                    return True

        say("please_confirm_retry")
    return False


def listen_digits(prompt_key, expected_len=None, max_wait=12, silence_end=2.0, attempts=2):
    """Speaks the prompt and listens for spoken digits (PIN / account
    number). Lenient: an exact-length match is used immediately; a longer
    string is trimmed; otherwise the best available guess is used."""
    for attempt in range(attempts):
        say(prompt_key)
        alternatives, _top = listen(max_wait=max_wait, max_total=15, silence_end=silence_end,
                                     return_alternatives=True, extra_langs=fallback_stt_langs(),
                                     hint="digits")
        if expected_len is None:
            for alt in alternatives:
                candidate = extract_numbers(alt)
                if candidate:
                    return candidate
        else:
            best = ""
            for alt in alternatives:
                candidate = extract_numbers(alt)
                if not candidate:
                    continue
                if len(candidate) == expected_len:
                    return candidate
                if len(candidate) > expected_len:
                    best = candidate[:expected_len]
                elif not best:
                    best = candidate
            if best:
                return best
        if attempt < attempts - 1:
            say("digits_not_clear")
    return ""


# ======================= CAMERA & FACE =======================

def _try_open(index, backend_name, backend_flag):
    cap = cv2.VideoCapture(index, backend_flag) if backend_flag is not None else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        print(f"[Camera] index {index} via {backend_name}: did not open")
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    good_frames = 0
    for _ in range(CAMERA_WARMUP_READS):
        ok, frame = cap.read()
        if ok and frame is not None:
            good_frames += 1
        time.sleep(0.03)

    if good_frames == 0:
        cap.release()
        print(f"[Camera] index {index} via {backend_name}: opened but delivered no frames")
        return None

    print(f"[Camera] index {index} via {backend_name}: OK ({good_frames}/{CAMERA_WARMUP_READS} warm-up frames)")
    return cap


def open_camera():
    for index in CAMERA_INDICES_TO_TRY:
        for backend_name, backend_flag in CAMERA_BACKENDS:
            cap = _try_open(index, backend_name, backend_flag)
            if cap is not None:
                return cap
    print("[Camera] All backend/index combinations failed.")
    return None


def test_camera():
    cap = open_camera()
    if cap is None:
        print("[Camera Test] FAILED - see reasons above.")
        return False
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        print(f"[Camera Test] SUCCESS - frame size {frame.shape[1]}x{frame.shape[0]}")
        return True
    print("[Camera Test] Opened but could not read a frame on final check.")
    return False


def _to_full(boxes, scale):
    return [(int(x / scale), int(y / scale), int(w / scale), int(h / scale)) for (x, y, w, h) in boxes]


def find_faces(frame):
    h, w = frame.shape[:2]
    scale = DETECT_WIDTH / float(w) if w > DETECT_WIDTH else 1.0
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale != 1.0 else frame
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    sw = gray.shape[1]

    frontal = frontal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(frontal) > 0:
        return _to_full(frontal, scale), []

    profile = [tuple(b) for b in profile_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))]
    flipped = cv2.flip(gray, 1)
    for (x, y, bw, bh) in profile_cascade.detectMultiScale(flipped, 1.1, 5, minSize=(60, 60)):
        profile.append((sw - (x + bw), y, bw, bh))
    return [], _to_full(profile, scale)


def largest_box(boxes):
    return max(boxes, key=lambda b: b[2] * b[3])


def guidance_for(box, frame_shape):
    """Returns a MESSAGES key if the face is badly placed, otherwise None."""
    x, y, w, h = box
    fh, fw = frame_shape[:2]
    cx = (x + w / 2) / fw
    cy = (y + h / 2) / fh
    size = w / fw
    if size < 0.12:
        return "move_closer"
    if size > 0.55:
        return "move_back"
    if cx > 0.65:
        return "move_right"
    if cx < 0.35:
        return "move_left"
    if cy < 0.25:
        return "lower_head"
    if cy > 0.70:
        return "raise_head"
    return None


def crop_face(frame, box, margin=0.35):
    x, y, w, h = box
    fh, fw = frame.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(fw, x + w + mx), min(fh, y + h + my)
    return frame[y0:y1, x0:x1]


def sharpness(frame, box):
    gray = cv2.cvtColor(crop_face(frame, box), cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def capture_face_image(save_path, timeout=CAMERA_TIMEOUT):
    holder = {}

    def opener():
        holder['cap'] = open_camera()

    opener_thread = threading.Thread(target=opener)
    opener_thread.start()
    say("face_opening")
    opener_thread.join()

    cap = holder.get('cap')
    if cap is None:
        say("camera_unavailable")
        return False

    winsound.Beep(1200, 200)

    chosen = None
    best_seen = None
    start = time.time()
    last_guidance = time.time() - 2.0
    consecutive_read_failures = 0

    try:
        while time.time() - start < timeout:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures > 30:
                    say("camera_lost_feed")
                    return False
                continue
            consecutive_read_failures = 0

            frontal, profile = find_faces(frame)
            now = time.time()

            if frontal:
                box = largest_box(frontal)
                if best_seen is None or box[2] > best_seen[1][2]:
                    best_seen = (frame.copy(), box)
                tip_key = guidance_for(box, frame.shape)
                if tip_key is None:
                    chosen = (frame, box)
                    break
                if now - last_guidance > GUIDANCE_GAP:
                    say(tip_key)
                    for _ in range(4):
                        cap.grab()
                    last_guidance = time.time()
            elif now - last_guidance > GUIDANCE_GAP:
                if profile:
                    say("face_side_profile")
                else:
                    say("face_not_found")
                for _ in range(4):
                    cap.grab()
                last_guidance = time.time()

        if chosen is None:
            chosen = best_seen

        if chosen is None:
            say("could_not_see_face")
            return False

        best_score = sharpness(*chosen)
        best_frame, best_box = chosen
        for _ in range(5):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            f2, _p = find_faces(frame)
            if f2:
                b2 = largest_box(f2)
                s2 = sharpness(frame, b2)
                if s2 > best_score:
                    best_score, best_frame, best_box = s2, frame, b2
    finally:
        cap.release()

    face_only = crop_face(best_frame, best_box)
    cv2.imwrite(save_path, face_only, [cv2.IMWRITE_JPEG_QUALITY, 98])
    winsound.Beep(1600, 150)
    say("face_captured")
    return True


def prepare_face(gray_img):
    faces = frontal_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        gray_img = gray_img[y:y + h, x:x + w]
    face = cv2.resize(gray_img, (200, 200))
    return cv2.equalizeHist(face)


def verify_face(account_id):
    stored_face_path = os.path.join(FACES_DIR, f"user_{account_id}.jpg")
    if not os.path.exists(stored_face_path):
        say("no_face_registered")
        return False

    temp_login_path = f"temp_login_{account_id}.jpg"
    try:
        if not capture_face_image(temp_login_path) or not os.path.exists(temp_login_path):
            say("face_verification_failed_access")
            return False

        img1 = cv2.imread(stored_face_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(temp_login_path, cv2.IMREAD_GRAYSCALE)
        face1 = prepare_face(img1)
        face2 = prepare_face(img2)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train([face1], np.array([0], dtype=np.int32))
        _label, confidence = recognizer.predict(face2)
    finally:
        if os.path.exists(temp_login_path):
            os.remove(temp_login_path)

    print(f"[Face Match Confidence]: {confidence:.2f} (pass if under {FACE_MATCH_THRESHOLD})")
    if confidence < FACE_MATCH_THRESHOLD:
        say("face_verification_passed")
        return True
    say("face_verification_failed")
    return False


# ======================= VOICE BIOMETRICS (withdrawals) =======================

def extract_voice_features(audio_file_path):
    y, sr_rate = librosa.load(audio_file_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr_rate * 0.3:
        raise ValueError("Voice sample too short")

    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y / peak

    mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=20)
    mfccs = mfccs[1:]
    delta = librosa.feature.delta(mfccs)

    feat = np.concatenate([np.mean(mfccs, axis=1), np.mean(delta, axis=1)])
    std = np.std(feat)
    if std > 1e-6:
        feat = (feat - np.mean(feat)) / std
    return feat


def sample_energy(audio_file_path):
    y, _sr = librosa.load(audio_file_path, sr=16000)
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float32) ** 2)) * 32768)


def verify_speakers(registered_audio_path, input_audio_path):
    f1 = extract_voice_features(registered_audio_path)
    f2 = extract_voice_features(input_audio_path)
    distance = float(np.linalg.norm(f1 - f2))
    print(f"[Voice Distance]: {distance:.2f} (pass if under {VOICE_MAX_DISTANCE})")
    return distance < VOICE_MAX_DISTANCE, distance


def verify_withdrawal_voice(account_id, attempts=3):
    ref = os.path.join(VOICES_DIR, f"user_{account_id}.wav")
    if not os.path.exists(ref):
        say("no_voice_profile")
        return False

    sample = f"temp_withdraw_{account_id}.wav"
    for attempt in range(attempts):
        say("withdrawal_phrase_prompt")
        if not capture_voice_sample(sample, max_wait=10, max_total=8):
            say("voice_not_heard_louder")
            continue

        energy = sample_energy(sample)
        if energy < MIN_VOICE_ENERGY:
            print(f"[Voice Energy]: {energy:.0f} (too quiet, need at least {MIN_VOICE_ENERGY})")
            if os.path.exists(sample):
                os.remove(sample)
            say("voice_too_quiet")
            continue

        try:
            passed, dist = verify_speakers(ref, sample)
        except Exception as e:
            print(f"[Voice Verify Error]: {e}")
            say("recording_too_short")
            passed = False
        finally:
            if os.path.exists(sample):
                os.remove(sample)

        if passed:
            say("voice_verified")
            return True
        if attempt < attempts - 1:
            say("voice_mismatch_retry")

    say("voice_verification_failed")
    return False


# ======================= DATABASE =======================

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


# ======================= BANKING LOGIC =======================

def hash_pin(pin, salt):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 100_000)


def delete_account(account_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM transactions WHERE account_id = ?", (account_id,))
    cur.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
    conn.commit()
    conn.close()
    for p in (os.path.join(FACES_DIR, f"user_{account_id}.jpg"),
              os.path.join(VOICES_DIR, f"user_{account_id}.wav")):
        if os.path.exists(p):
            os.remove(p)


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
    face_ok = False
    for attempt in range(2):
        if capture_face_image(face_path):
            face_ok = True
            break
        if attempt == 0:
            say("let_try_camera_again")
    if not face_ok:
        delete_account(account_id)
        say("could_not_register_face")
        return None

    voice_path = os.path.join(VOICES_DIR, f"user_{account_id}.wav")
    voice_ok = False
    for _ in range(2):
        say("voice_registration_prompt")
        if capture_voice_sample(voice_path, max_wait=10, max_total=8):
            voice_ok = True
            break
        say("voice_not_heard")
    if not voice_ok:
        delete_account(account_id)
        say("could_not_register_voice")
        return None

    say("voice_registered")
    return account_id


def login(account_id, pin):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT salt, pin_hash FROM accounts WHERE id = ?", (account_id,))
    row = cur.fetchone()
    conn.close()
    if row is None:
        print(f"[Login] No account found with id {account_id}")
        return False
    salt, stored_hash = row
    if not hmac.compare_digest(hash_pin(pin, salt), stored_hash):
        print(f"[Login] PIN mismatch for account {account_id}")
        return False
    return verify_face(account_id)


def get_balance(account_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT balance FROM accounts WHERE id = ?", (account_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else 0.0


def record_transaction(cur, account_id, kind, amount):
    cur.execute(
        "INSERT INTO transactions (account_id, type, amount, timestamp) VALUES (?, ?, ?, ?)",
        (account_id, kind, amount, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )


def deposit(account_id, amount):
    if amount <= 0:
        return False, None
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE accounts SET balance = balance + ? WHERE id = ?", (amount, account_id))
    record_transaction(cur, account_id, "DEPOSIT", amount)
    conn.commit()
    conn.close()
    return True, get_balance(account_id)


def withdraw(account_id, amount):
    if amount <= 0:
        return False, None, "invalid"
    current_bal = get_balance(account_id)
    if amount > current_bal:
        return False, current_bal, "insufficient"
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE accounts SET balance = balance - ? WHERE id = ?", (amount, account_id))
    record_transaction(cur, account_id, "WITHDRAW", amount)
    conn.commit()
    conn.close()
    return True, get_balance(account_id), None


def play_cash_dispensing_sound():
    try:
        for i in range(DISPENSE_BEEP_COUNT):
            freq = DISPENSE_BEEP_FREQS[i % 2]
            winsound.Beep(freq, DISPENSE_BEEP_MS)
    except Exception as e:
        print(f"[Dispensing Sound Error]: {e}")


def mini_statement(account_id, limit=5):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT type, amount, timestamp FROM transactions WHERE account_id = ? ORDER BY id DESC LIMIT ?",
        (account_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return rows