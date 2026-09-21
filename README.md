# VoiceBank — Accessible Voice & Face-Verified Banking System

An offline, accessible Python banking system designed for visually impaired users. Features full voice navigation, voice command processing, facial recognition verification, and local transactional security.

## Features
- **Voice Guidance & Navigation:** Full text-to-speech audio feedback.
- **Biometric Security:** Facial verification using OpenCV.
- **Voice Commands:** Hands-free command parsing and digit processing.
- **Transaction Engine:** Local SQLite storage with secure PIN hashing.

## Setup & Installation

```bash
# Clone repository
git clone [https://github.com/dsaranya1219/VoiceBank.git](https://github.com/dsaranya1219/VoiceBank.git)
cd VoiceBank

# Install dependencies
pip install opencv-python pyttsx3 SpeechRecognition sounddevice numpy
