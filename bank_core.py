"""
VoiceBank - Core Module
Voice + face guided banking for blind and low-vision users.
"""

import sqlite3
import hashlib
import hmac
import os
import re
import time
import wave
import threading
import winsound
from datetime import datetime

import cv2
import numpy as np
import pyttsx3
import sounddevice as sd
import speech_recognition as sr
import librosa

try:
    import pythoncom
except ImportError:
    pythoncom = None

# ======================= SETTINGS (tune here) =======================
DB_NAME = "voicebank.db"
FACES_DIR = "face_data"
VOICES_DIR = "voice_data"

# Microphone / listening
SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.05
CALIBRATION_BLOCKS = 5
VAD_MIN_THRESHOLD = 55         # lowered so quiet/soft voices still trigger speech detection
VAD_MAX_THRESHOLD = 1500
VAD_NOISE_MULTIPLIER = 1.5     # lowered from 1.8 -> more sensitive to soft speech over room noise
PRE_ROLL_BLOCKS = 10
POST_ROLL_BLOCKS = 8           # a bit more tail room so soft trailing words aren't clipped
MIN_SPEECH_BLOCKS = 4          # ignore tiny blips (coughs, clicks) shorter than ~0.2s

# Camera
CAMERA_INDEX = 0
CAMERA_BACKENDS = [           # tried in this order; whichever actually delivers frames wins
    ("CAP_DSHOW", cv2.CAP_DSHOW),
    ("CAP_MSMF", cv2.CAP_MSMF),
    ("CAP_ANY", cv2.CAP_ANY),
]
CAMERA_INDICES_TO_TRY = [0, 1]   # falls back to index 1 if 0 is busy/unavailable
CAMERA_WARMUP_READS = 5          # frames we must successfully read before trusting the camera
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
DETECT_WIDTH = 640
CAMERA_TIMEOUT = 20
GUIDANCE_GAP = 4

# Verification
FACE_MATCH_THRESHOLD = 65
VOICE_MAX_DISTANCE = 40.0      # voice: smaller distance = more similar (forgiving threshold)
MIN_VOICE_ENERGY = 150         # below this average loudness, treat sample as "too quiet/unclear"

# Cash dispensing sound (plays right when a withdrawal succeeds)
DISPENSE_BEEP_COUNT = 6
DISPENSE_BEEP_MS = 90
DISPENSE_BEEP_FREQS = (750, 1050)   # alternates between these two tones

for _path in [FACES_DIR, VOICES_DIR]:
    if not os.path.exists(_path):
        os.makedirs(_path)

frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

speak_lock = threading.Lock()


# ======================= SPEECH OUTPUT =======================

def speak(text):
    """Prints and speaks text through the speakers. Only one speak() runs at a time app-wide."""
    print(f"\n[VoiceBank]: {text}")
    with speak_lock:
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


# ======================= SMART LISTENING =======================

# Lets a manual button click interrupt an in-progress voice listen, so the
# dashboard can offer "click or speak, whichever you prefer" like the front page.
cancel_listen_event = threading.Event()


def request_cancel_listen():
    """Called when the user clicks a manual button; breaks an active listen() out early."""
    cancel_listen_event.set()


def record_until_silence(max_wait=10, max_total=30, silence_end=1.6):
    """
    Records from the microphone. Waits up to `max_wait` seconds for the person to start
    speaking, then keeps recording while they speak and stops once they have been silent
    for `silence_end` seconds. Tuned to be forgiving of quiet/soft speech.
    Returns a numpy int16 array, or None if nobody spoke or listening was cancelled.
    """
    cancel_listen_event.clear()
    block_size = int(SAMPLE_RATE * BLOCK_SECONDS)
    blocks = []
    noise_samples = []
    threshold = VAD_MIN_THRESHOLD
    speech_started = False
    consecutive_voiced = 0
    silent_run = 0
    first_voiced = None
    last_voiced = None
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

            if idx < CALIBRATION_BLOCKS:
                noise_samples.append(rms)
                if idx == CALIBRATION_BLOCKS - 1:
                    noise = float(np.median(noise_samples))
                    threshold = min(max(noise * VAD_NOISE_MULTIPLIER, VAD_MIN_THRESHOLD), VAD_MAX_THRESHOLD)
                    print(f"[Mic] noise level {noise:.0f}, speech threshold {threshold:.0f}")
                continue

            # Once speech has started, drop the bar further so soft/trailing words
            # (low volume endings, breathy speech) are not mistaken for silence.
            limit = threshold * 0.45 if speech_started else threshold
            voiced = rms > limit

            if voiced:
                consecutive_voiced += 1
                silent_run = 0
                last_voiced = idx
                if not speech_started and consecutive_voiced >= 1:
                    speech_started = True
                    first_voiced = idx - 1
            else:
                consecutive_voiced = 0
                if speech_started:
                    silent_run += 1

            if speech_started and silent_run * BLOCK_SECONDS >= silence_end:
                break
            if not speech_started and elapsed >= max_wait:
                return None
            if elapsed >= max_total:
                break

    if not speech_started:
        return None
    if (last_voiced - first_voiced) < MIN_SPEECH_BLOCKS:
        return None  # too short to be real speech (a click, a cough, mic noise)

    start_i = max(0, first_voiced - PRE_ROLL_BLOCKS)
    end_i = min(len(blocks), last_voiced + POST_ROLL_BLOCKS + 1)
    return np.concatenate(blocks[start_i:end_i])


def save_wav(path, audio):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def transcribe(path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(path) as source:
            audio = recognizer.record(source)
        return recognizer.recognize_google(audio).strip().lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        speak("Speech recognition service is unreachable. Please check your internet connection.")
        return ""


def listen(max_wait=10, max_total=30, silence_end=1.6, save_voice_path=None):
    """
    Beeps, listens until the person stops speaking, and returns the recognized text.
    Returns "" if nothing was heard or understood.
    """
    filename = save_voice_path if save_voice_path else "temp_voice.wav"
    print("\n[VoiceBank is listening... speak after the beep]")
    winsound.Beep(1000, 250)
    try:
        audio = record_until_silence(max_wait=max_wait, max_total=max_total, silence_end=silence_end)
        if audio is None:
            if cancel_listen_event.is_set():
                return ""  # cancelled by a manual button click; stay quiet, don't announce "I did not hear anything"
            speak("I did not hear anything.")
            return ""
        save_wav(filename, audio)
        text = transcribe(filename)
        if text:
            print(f"[Recognized Speech]: {text}")
        else:
            speak("Sorry, I could not understand what you said.")
        return text
    except Exception as e:
        print(f"[Voice Capture Notice]: {e}")
        return ""
    finally:
        if not save_voice_path and os.path.exists(filename):
            try:
                os.remove(filename)
            except Exception:
                pass


def capture_voice_sample(path, max_wait=10, max_total=12):
    """Records a voice sample for biometrics (no transcription). Returns True if saved.
    max_total raised to 12s so a slower/quieter speaker still gets a full sample."""
    print("\n[VoiceBank is listening for your voice sample... speak after the beep]")
    winsound.Beep(1000, 250)
    try:
        audio = record_until_silence(max_wait=max_wait, max_total=max_total, silence_end=1.6)
        if audio is None:
            return False
        save_wav(path, audio)
        return True
    except Exception as e:
        print(f"[Voice Sample Error]: {e}")
        return False


# ======================= SPOKEN NUMBERS, AMOUNTS, COMMANDS =======================

def extract_numbers(text):
    """Spoken digits to a digit string. Example: 'one two three four' -> '1234'."""
    if not text:
        return ""
    word_to_num = {
        'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'to': '2', 'too': '2',
        'three': '3', 'four': '4', 'for': '4', 'five': '5', 'six': '6',
        'seven': '7', 'eight': '8', 'ate': '8', 'nine': '9'
    }
    converted = ""
    for w in text.lower().split():
        if w in word_to_num:
            converted += word_to_num[w]
        else:
            converted += "".join(c for c in w if c.isdigit())
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
    """Understands '5000', '5 thousand', 'two thousand five hundred', '2.5 lakh'. Returns float or None."""
    if not text:
        return None
    t = text.lower().replace(',', '')
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


COMMANDS = [
    ('logout', ['log out', 'logout', 'sign out', 'exit', 'quit', 'goodbye', 'bye']),
    ('balance', ['balance']),
    ('withdraw', ['withdraw', 'withdrawal', 'take out', 'take money', 'debit']),
    ('deposit', ['deposit', 'add money', 'put money', 'credit']),
    ('statement', ['statement', 'transaction', 'history', 'recent']),
]


def parse_command(text):
    """Returns 'balance', 'deposit', 'withdraw', 'statement', 'logout', 'unknown' or 'none' (nothing heard)."""
    if not text:
        return 'none'
    t = text.lower()
    for action, keys in COMMANDS:
        if any(k in t for k in keys):
            return action
    return 'unknown'


YES_WORDS = {'yes', 'yeah', 'yep', 'yup', 'confirm', 'correct', 'right', 'ok', 'okay', 'sure', 'proceed'}
NO_WORDS = {'no', 'nope', 'cancel', 'stop', 'wrong', 'negative', 'incorrect'}


def ask_yes_no(prompt, attempts=3):
    """Asks a question by voice and returns True (confirmed) or False (rejected / unclear)."""
    for _ in range(attempts):
        speak(prompt)
        heard = listen(max_wait=8, max_total=14, silence_end=1.3)
        words = set(heard.replace(',', ' ').split())
        if words & NO_WORDS:
            return False
        if words & YES_WORDS:
            return True
        if heard:
            speak("Please say correct, or wrong.")
    return False


# ======================= CAMERA & FACE =======================

def _try_open(index, backend_name, backend_flag):
    """Opens one specific index+backend combo and proves it actually delivers frames,
    not just that isOpened() says True (which can lie)."""
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
        print(f"[Camera] index {index} via {backend_name}: opened but delivered no frames "
              f"(likely in use by another app, or driver issue)")
        return None

    print(f"[Camera] index {index} via {backend_name}: OK ({good_frames}/{CAMERA_WARMUP_READS} warm-up frames)")
    return cap


def open_camera():
    """
    Opens the webcam robustly: tries each backend on each index until one actually
    delivers frames (not just reports 'opened'). Prints exactly what it tried so a
    failure can be diagnosed from the terminal instead of guessed at.
    """
    for index in CAMERA_INDICES_TO_TRY:
        for backend_name, backend_flag in CAMERA_BACKENDS:
            cap = _try_open(index, backend_name, backend_flag)
            if cap is not None:
                return cap
    print("[Camera] All backend/index combinations failed. The camera may be in use by "
          "another application (close Zoom/Teams/other camera apps), disabled in Windows "
          "privacy settings, or CAMERA_INDEX/CAMERA_INDICES_TO_TRY needs adjusting.")
    return None


def test_camera():
    """Standalone diagnostic. Run: python -c \"import bank_core; bank_core.test_camera()\"
    Opens the camera, shows what backend worked, and reports a live frame size."""
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
    """Returns (frontal_boxes, profile_boxes) in full-resolution coordinates."""
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
    """Returns a spoken instruction if the face is badly placed, otherwise None."""
    x, y, w, h = box
    fh, fw = frame_shape[:2]
    cx = (x + w / 2) / fw
    cy = (y + h / 2) / fh
    size = w / fw
    if size < 0.12:
        return "Please move a little closer to the camera."
    if size > 0.55:
        return "Please move a little back from the camera."
    if cx > 0.65:
        return "Please move a little to your right."
    if cx < 0.35:
        return "Please move a little to your left."
    if cy < 0.25:
        return "Please lower your head a little."
    if cy > 0.70:
        return "Please raise your head a little."
    return None


def crop_face(frame, box, margin=0.35):
    """Crops just the face (with a small margin) so the background is not saved."""
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
    """
    Opens the camera robustly, guides the person by voice, and saves a sharp crop of the face.
    """
    holder = {}

    def opener():
        holder['cap'] = open_camera()

    opener_thread = threading.Thread(target=opener)
    opener_thread.start()
    speak("Opening the camera. Please face the screen and keep your head still. I will guide you.")
    opener_thread.join()

    cap = holder.get('cap')
    if cap is None:
        speak("The camera is unavailable. Please close any other app that might be using it, "
              "such as Zoom, Teams, or another browser tab, and try again.")
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
                    print("[Camera] Lost the video feed mid-capture (repeated read failures).")
                    speak("I lost the camera feed. Please try again.")
                    return False
                continue
            consecutive_read_failures = 0

            frontal, profile = find_faces(frame)
            now = time.time()

            if frontal:
                box = largest_box(frontal)
                if best_seen is None or box[2] > best_seen[1][2]:
                    best_seen = (frame.copy(), box)
                tip = guidance_for(box, frame.shape)
                if tip is None:
                    chosen = (frame, box)
                    break
                if now - last_guidance > GUIDANCE_GAP:
                    speak(tip)
                    for _ in range(4):
                        cap.grab()
                    last_guidance = time.time()
            elif now - last_guidance > GUIDANCE_GAP:
                if profile:
                    speak("I can see the side of your face. Please turn your face toward the camera.")
                else:
                    speak("I cannot see your face. Please look toward the screen and move slowly until you hear a beep.")
                for _ in range(4):
                    cap.grab()
                last_guidance = time.time()

        if chosen is None:
            chosen = best_seen

        if chosen is None:
            speak("I could not see your face. Please check the lighting and try again.")
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
    speak("Face captured clearly.")
    return True


def prepare_face(gray_img):
    """Finds the face in a grayscale image, crops it, resizes and evens out lighting."""
    faces = frontal_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        gray_img = gray_img[y:y + h, x:x + w]
    face = cv2.resize(gray_img, (200, 200))
    return cv2.equalizeHist(face)


def verify_face(account_id):
    """Compares a live face against the registered one using LBPH."""
    stored_face_path = os.path.join(FACES_DIR, f"user_{account_id}.jpg")
    if not os.path.exists(stored_face_path):
        speak("No registered face was found for this account. Access denied.")
        return False

    temp_login_path = f"temp_login_{account_id}.jpg"
    try:
        if not capture_face_image(temp_login_path) or not os.path.exists(temp_login_path):
            speak("Face verification failed. Access denied.")
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
        speak("Face verification passed.")
        return True
    speak("Face verification failed. This does not match the registered face.")
    return False


# ======================= VOICE BIOMETRICS (withdrawals) =======================
# This compares the actual sound of the voice (pitch, tone texture) - it never
# looks at the words that were said, so it verifies the SPEAKER, not the phrase.

def extract_voice_features(audio_file_path):
    """Builds a voice-print: average MFCCs plus their rate of change (deltas),
    which makes the print steadier across different recording sessions and less
    sensitive to background noise / mic gain differences."""
    y, sr_rate = librosa.load(audio_file_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) < sr_rate * 0.3:
        raise ValueError("Voice sample too short")

    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 0:
        y = y / peak

    mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=20)
    mfccs = mfccs[1:]  # skip coefficient 0 (just overall loudness)
    delta = librosa.feature.delta(mfccs)

    feat = np.concatenate([np.mean(mfccs, axis=1), np.mean(delta, axis=1)])
    std = np.std(feat)
    if std > 1e-6:
        feat = (feat - np.mean(feat)) / std
    return feat


def sample_energy(audio_file_path):
    """Average loudness of a saved wav, used to detect 'too quiet to verify' samples."""
    y, _sr = librosa.load(audio_file_path, sr=16000)
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(y.astype(np.float32) ** 2)) * 32768)


def verify_speakers(registered_audio_path, input_audio_path):
    """Returns (passed, distance). Smaller distance = more similar voices.
    This is a biometric comparison of voice characteristics, never a transcript check."""
    f1 = extract_voice_features(registered_audio_path)
    f2 = extract_voice_features(input_audio_path)
    distance = float(np.linalg.norm(f1 - f2))
    print(f"[Voice Distance]: {distance:.2f} (pass if under {VOICE_MAX_DISTANCE})")
    return distance < VOICE_MAX_DISTANCE, distance


def verify_withdrawal_voice(account_id, attempts=3):
    ref = os.path.join(VOICES_DIR, f"user_{account_id}.wav")
    if not os.path.exists(ref):
        speak("No voice profile is registered for this account, so withdrawals are not allowed. Please create a new account.")
        return False

    sample = f"temp_withdraw_{account_id}.wav"
    for attempt in range(attempts):
        speak("For withdrawal security, after the beep, please say clearly: my voice is my password.")
        if not capture_voice_sample(sample, max_wait=10, max_total=12):
            speak("I did not hear you. Please speak a little louder and closer to the microphone.")
            continue

        energy = sample_energy(sample)
        if energy < MIN_VOICE_ENERGY:
            print(f"[Voice Energy]: {energy:.0f} (too quiet, need at least {MIN_VOICE_ENERGY})")
            if os.path.exists(sample):
                os.remove(sample)
            speak("That was too quiet for me to verify reliably. Please speak a bit louder.")
            continue

        try:
            passed, dist = verify_speakers(ref, sample)
        except Exception as e:
            print(f"[Voice Verify Error]: {e}")
            speak("That recording was too short or unclear. Please try again.")
            passed = False
        finally:
            if os.path.exists(sample):
                os.remove(sample)

        if passed:
            speak("Voice verified.")
            return True
        if attempt < attempts - 1:
            speak("Your voice did not match closely enough. Let's try once more, speak clearly and steadily.")

    speak("Voice verification failed. Withdrawal cancelled.")
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
    """Creates the account, registers face and voice. Returns account id, or None on failure."""
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
            speak("Let's try the camera once more.")
    if not face_ok:
        delete_account(account_id)
        speak("I could not register your face, so the account was not created. Please try again.")
        return None

    voice_path = os.path.join(VOICES_DIR, f"user_{account_id}.wav")
    voice_ok = False
    for _ in range(2):
        speak("Now let's register your voice for withdrawal security. After the beep, say clearly: my voice is my password.")
        if capture_voice_sample(voice_path, max_wait=10, max_total=12):
            voice_ok = True
            break
        speak("I did not hear you.")
    if not voice_ok:
        delete_account(account_id)
        speak("I could not register your voice, so the account was not created. Please try again.")
        return None

    speak("Voice registered successfully.")
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
        print(f"[Login] PIN mismatch for account {account_id} (entered pin: {pin})")
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


def play_cash_dispensing_sound():
    """
    Plays a short mechanical-sounding cue so the user has clear audible confirmation
    that cash is being released. Runs synchronously (blocking, ~0.5s) and on purpose:
    it should finish playing BEFORE the spoken confirmation starts, so the two sounds
    don't overlap and the sequence is unambiguous - dispensing sound, then "please
    collect your cash..." message.
    """
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