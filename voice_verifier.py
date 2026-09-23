import numpy as np
import librosa

def extract_voice_features(audio_file_path):
    """
    Extracts MFCC features (spectral voice characteristics) from an audio file.
    """
    # Load audio at 16kHz
    y, sr = librosa.load(audio_file_path, sr=16000)
    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Average across time to get a single feature vector representing the speaker's voice
    mean_mfccs = np.mean(mfccs.T, axis=0)
    return mean_mfccs

def verify_speakers(registered_audio_path, input_audio_path, threshold=0.80):
    """
    Compares two voice samples and returns True if they belong to the same person.
    """
    feat1 = extract_voice_features(registered_audio_path)
    feat2 = extract_voice_features(input_audio_path)
    
    # Calculate cosine similarity between speaker feature vectors
    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
    
    print(f"Voice Similarity Score: {similarity:.4f}")
    return similarity >= threshold, float(similarity)

if __name__ == "__main__":
    print("Voice Verifier Module Loaded Successfully!")