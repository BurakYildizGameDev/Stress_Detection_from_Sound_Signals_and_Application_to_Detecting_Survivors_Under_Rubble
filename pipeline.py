import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import librosa
import joblib

from features import FEATURE_FUNCS, FEATURE_SPECS, HUMAN_SR

# Temel Yapılandırma
SR = HUMAN_SR
WINDOW_SEC = 1.0
HOP_SEC = 0.5
RMS_THRESHOLD = 0.01
HUMAN_THRESHOLD = 0.20
EMERGENCY_THRESHOLD = 0.50

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "human": "human_detector_mid.pkl",
    "emergency": "emergency_classifier.pkl",
}

# Metadata'sız, tamsayı etiketle kaydedilmiş eski modeller için
LEGACY_LABELS = {
    "human": {0: "non_human", 1: "human"},
    "emergency": {0: "normal", 1: "stress", 2: "scream", 3: "panic"},
}


class ModelLoadError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    task: str
    model: object
    labels: list        # model.classes_ sırasıyla
    metadata: dict      # eski modellerde boş

    def proba(self, feats):
        return self.model.predict_proba(feats.reshape(1, -1))[0]


def metadata_path(model_path):
    return os.path.splitext(model_path)[0] + ".json"


def load_model(task, models_dir=MODELS_DIR):
    """Modeli yükler ve özellik tanımıyla uyumunu doğrular."""
    path = os.path.join(models_dir, MODEL_FILES[task])
    rel = os.path.relpath(path, BASE_DIR)
    train_hint = f"python scripts/train.py --task {task}"

    if not os.path.exists(path):
        raise ModelLoadError(
            f"{task} modeli bulunamadı: {rel}\n"
            f"`git lfs pull` ile indirin ya da `{train_hint}` ile eğitin.")
    with open(path, "rb") as f:
        if f.read(64).startswith(b"version https://git-lfs"):
            raise ModelLoadError(
                f"{rel} bir Git LFS işaretçisi, model içeriği indirilmemiş.\n"
                "Çalıştırın: git lfs install && git lfs pull")

    spec = FEATURE_SPECS[task]
    metadata = {}
    if os.path.exists(metadata_path(path)):
        with open(metadata_path(path), encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("feature_spec") != spec:
            raise ModelLoadError(
                f"{rel} farklı bir öznitelik tanımıyla eğitilmiş.\n"
                f"  model: {metadata.get('feature_spec')}\n"
                f"  kod:   {spec}\n"
                f"Modeli yeniden eğitin: {train_hint}")
        import sklearn
        trained_with = metadata.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            warnings.warn(f"{rel} scikit-learn {trained_with} ile kaydedilmiş, "
                          f"kurulu sürüm {sklearn.__version__}. Tahminler farklı olabilir.")

    try:
        model = joblib.load(path)
    except Exception as e:
        raise ModelLoadError(f"{rel} yüklenemedi: {e}") from e

    n = getattr(model, "n_features_in_", None)
    if n != spec["n_features"]:
        raise ModelLoadError(
            f"{rel} {n} öznitelik bekliyor, features.py {spec['n_features']} üretiyor.\n"
            f"Modeli yeniden eğitin: {train_hint}")

    classes = list(model.classes_)
    if all(isinstance(c, str) for c in classes):
        labels = [str(c) for c in classes]
    else:
        legacy = LEGACY_LABELS[task]
        labels = [legacy[int(c)] for c in classes]
    return LoadedModel(task, model, labels, metadata)


_models = {}


def get_model(task):
    if task not in _models:
        _models[task] = load_model(task)
    return _models[task]


# =========================
# CORE PIPELINE
# =========================
def analyze_audio_array(y, sr=SR):
    rms_val = float(np.mean(librosa.feature.rms(y=y)))
    if rms_val < RMS_THRESHOLD:
        return {"status": "silence", "rms": rms_val}

    # 1. Aşama: İnsan sesi var mı?
    human = get_model("human")
    human_probs = human.proba(FEATURE_FUNCS["human"](y, sr=sr))
    human_prob = float(human_probs[human.labels.index("human")])

    if human_prob < HUMAN_THRESHOLD:
        return {
            "status": "no_human",
            "human_prob": human_prob,
            "rms": rms_val
        }

    # 2. Aşama: Acil durum / Stres sınıflandırma
    emergency = get_model("emergency")
    probs = emergency.proba(FEATURE_FUNCS["emergency"](y, sr=sr))

    idx = int(np.argmax(probs))
    state = emergency.labels[idx]
    state_confidence = float(probs[idx])

    # Acil durum olasılığı: normal dışındaki sınıfların toplamı
    emergency_prob = float(1.0 - probs[emergency.labels.index("normal")])
    is_emergency = (state != "normal") and (emergency_prob >= EMERGENCY_THRESHOLD)

    return {
        "status": "DETECTED",
        "state": state,
        "is_emergency": is_emergency,
        "human_prob": human_prob,
        "state_confidence": state_confidence,
        "emergency_prob": emergency_prob,
        "rms": rms_val,
        "all_probs": {label: float(p) for label, p in zip(emergency.labels, probs)}
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
