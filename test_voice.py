import sounddevice as sd
import numpy as np
import speech_recognition as sr
import scipy.io.wavfile as wav
import os

def record_audio(filename="temp.wav", duration=5, sample_rate=44100):
    print("Listening... Speak into your microphone now.")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, sample_rate, recording)

def recognize_speech():
    record_audio(duration=4)
    recognizer = sr.Recognizer()
    
    with sr.AudioFile("temp.wav") as source:
        audio = recognizer.record(source)
        
    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print(f"Service error: {e}")
    finally:
        if os.path.exists("temp.wav"):
            os.remove("temp.wav")

if __name__ == "__main__":
    recognize_speech()