import numpy as np
import librosa
import joblib
import sounddevice as sd
import queue

SR = 22050
WINDOW_SEC = 1.0
HOP_SEC = 0.5
RMS_THRESHOLD = 0.01

human_model = joblib.load("models/human_detector_mid.pkl")
emergency_model = joblib.load("models/emergency_classifier.pkl")

LABELS = ["normal", "stress", "scream", "panic"]

# =========================
# FEATURE EXTRACTORS
# =========================
def extract_human_features(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    feats = []
    feats.extend(np.mean(mfcc, axis=1))   # 13
    feats.extend(np.std(mfcc, axis=1))    # 13
    feats.append(np.mean(librosa.feature.zero_crossing_rate(y)))  # 1
    feats.append(np.mean(librosa.feature.rms(y=y)))               # 1
    return np.array(feats).reshape(1, -1)  # 28

def extract_emergency_features(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13)
    feats = []
    feats.extend(np.mean(mfcc, axis=1))   # 13
    feats.append(np.mean(librosa.feature.zero_crossing_rate(y)))  # 1
    feats.append(np.mean(librosa.feature.rms(y=y)))               # 1
    return np.array(feats).reshape(1, -1)  # 15

# =========================
# CORE PIPELINE
# =========================
def analyze_audio_array(y):
    rms = np.mean(librosa.feature.rms(y=y))
    if rms < RMS_THRESHOLD:
        return {"status": "silence"}

    human_feat = extract_human_features(y)
    human_prob = human_model.predict_proba(human_feat)[0][1]

    if human_prob < 0.15:
        return {
            "status": "no_human",
            "human_prob": float(human_prob)
        }

    emergency_feat = extract_emergency_features(y)
    probs = emergency_model.predict_proba(emergency_feat)[0]

    idx = int(np.argmax(probs))
    return {
        "status": "DETECTED",
        "state": LABELS[idx],
        "human_prob": float(human_prob),
        "emergency_prob": float(probs[idx])
    }

# =========================
# FILE MODE
# =========================
def analyze_file(path):
    y, _ = librosa.load(path, sr=SR)
    return analyze_audio_array(y)

