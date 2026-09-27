import os
import numpy as np
import librosa
import joblib

from features import HUMAN_SR, human_features, emergency_features

# Temel Yapılandırma
SR = HUMAN_SR
WINDOW_SEC = 1.0
HOP_SEC = 0.5
RMS_THRESHOLD = 0.01

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

human_model_path = os.path.join(MODELS_DIR, "human_detector_mid.pkl")
emergency_model_path = os.path.join(MODELS_DIR, "emergency_classifier.pkl")

human_model = joblib.load(human_model_path)
emergency_model = joblib.load(emergency_model_path)

LABELS = ["normal", "stress", "scream", "panic"]

# =========================
# FEATURE EXTRACTORS
# =========================
def extract_human_features(y, sr=SR):
    return human_features(y, sr=sr).reshape(1, -1)

def extract_emergency_features(y, sr=SR):
    return emergency_features(y, sr=sr).reshape(1, -1)

# =========================
# CORE PIPELINE
# =========================
def analyze_audio_array(y, sr=SR):
    rms_val = float(np.mean(librosa.feature.rms(y=y)))
    if rms_val < RMS_THRESHOLD:
        return {"status": "silence", "rms": rms_val}

    # 1. Aşama: İnsan sesi var mı?
    human_feat = extract_human_features(y, sr=sr)
    human_prob = float(human_model.predict_proba(human_feat)[0][1])

    if human_prob < 0.20:
        return {
            "status": "no_human",
            "human_prob": human_prob,
            "rms": rms_val
        }

    # 2. Aşama: Acil durum / Stres sınıflandırma
    emergency_feat = extract_emergency_features(y, sr=sr)
    probs = emergency_model.predict_proba(emergency_feat)[0]

    idx = int(np.argmax(probs))
    state = LABELS[idx]
    state_confidence = float(probs[idx])

    # Acil durum olasılığı: normal dışındaki durumların (stres, çığlık, panik) toplamı
    emergency_prob = float(1.0 - probs[0]) if len(probs) > 0 else 0.0
    is_emergency = (state != "normal") and (emergency_prob >= 0.50)

    return {
        "status": "DETECTED",
        "state": state,
        "is_emergency": is_emergency,
        "human_prob": human_prob,
        "state_confidence": state_confidence,
        "emergency_prob": emergency_prob,
        "all_probs": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    }

# =========================
# FILE MODE
# =========================
def analyze_file(path):
    y, sr = librosa.load(path, sr=SR)
    return analyze_audio_array(y, sr=sr)

# Geriye dönük uyumluluk için alias
def analyze_audio(path):
    return analyze_file(path)
