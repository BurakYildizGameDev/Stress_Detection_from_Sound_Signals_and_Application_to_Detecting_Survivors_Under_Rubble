"""
pipeline_v2.py - Fısıltı, İnleme ve 5 Sınıflı Acil Durum Çok Kademeli Canlı Karar Hattı.

Bu modül, arama kurtarma ses analizini 3 kritik yenilikle yönetir:

1. Düşük Enerjili Fısıltı Kapısı (Whisper RMS Gate):
   - Klasik sistemler RMS < 0.01 sesleri sessizlik sayarak eler.
   - Enkazın 48-72. saatlerinde fısıltı ve stridor genliği 0.002 - 0.008 seviyesindedir.
   - Eşik değeri 0.0015'e çekilmiştir.
2. 5 Sınıflı Kademeli Karar Mekanizması:
   - Aşama 1: 34 boyutlu öznitelik ile insan sesi doğrulaması (P >= 0.20).
   - Aşama 2: 20 boyutlu öznitelik ile ('normal', 'stress', 'panic', 'moan', 'whisper') sınıflandırması.
3. Öncelikli Alarm Seviyeleri (Priority Triage):
   - CRITICAL_SURVIVOR: Fısıltı veya inleme (Yaşam belirtisi çok zayıf, acil müdahale!).
   - HIGH_EMERGENCY: Çığlık, panik veya yüksek stres.
   - NORMAL_SPEECH: Sakin konuşma / stabil kazazede.
"""
import os
import json
import numpy as np
import librosa
import joblib

from features import HUMAN_SR, EMERGENCY_SR
from features_v2 import (
    human_features_v2,
    emergency_features_v2,
    FEATURE_SPECS_V2,
    FEATURE_FUNCS_V2
)

# Yapılandırma
SR = HUMAN_SR
WINDOW_SEC = 1.0
RMS_SILENCE_THRESHOLD = 0.0015
# Model meta verisinde eşik yoksa kullanılır. train_v2.py eşiği doğrulama
# verisinde yanlış alarm <= %10 olacak şekilde seçip .json'a yazar.
HUMAN_PROB_THRESHOLD = 0.50
EMERGENCY_PROB_THRESHOLD = 0.45

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES_V2 = {
    "human": "human_detector_v2.pkl",
    "emergency": "emergency_classifier_v2.pkl",
}

# Öncelik seviyeleri
ALERT_PRIORITY = {
    "whisper": "CRITICAL_SURVIVOR",
    "moan": "CRITICAL_SURVIVOR",
    "panic": "HIGH_EMERGENCY",
    "stress": "HIGH_EMERGENCY",
    "normal": "NORMAL_SPEECH"
}


class PipelineV2:
    def __init__(self, human_model=None, emergency_model=None, models_dir=MODELS_DIR,
                 human_threshold=HUMAN_PROB_THRESHOLD):
        self.models_dir = models_dir
        self.human_threshold = human_threshold
        self.human_model = human_model
        self.emergency_model = emergency_model
        self._load_if_needed()

    def _load_if_needed(self):
        if self.human_model is None:
            path_h = os.path.join(self.models_dir, MODEL_FILES_V2["human"])
            if os.path.exists(path_h):
                self.human_model = joblib.load(path_h)
                meta_path = os.path.splitext(path_h)[0] + ".json"
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as f:
                        self.human_threshold = json.load(f).get("human_threshold") or HUMAN_PROB_THRESHOLD

        if self.emergency_model is None:
            path_e = os.path.join(self.models_dir, MODEL_FILES_V2["emergency"])
            if os.path.exists(path_e):
                self.emergency_model = joblib.load(path_e)

    def analyze_audio_array(self, y, sr=SR):
        """
        1 saniyelik ham ses sinyali üzerinde kademeli analiz yapar.
        """
        rms_val = float(np.mean(librosa.feature.rms(y=y)))
        if rms_val < RMS_SILENCE_THRESHOLD:
            return {
                "status": "silence",
                "rms": rms_val,
                "is_emergency": False,
                "priority": "NONE"
            }

        # Model yüklü değilse (eğitim öncesi fallback)
        if self.human_model is None or self.emergency_model is None:
            # Enerji varsa ama model yoksa temel tespit dön
            return {
                "status": "RAW_AUDIO",
                "rms": rms_val,
                "is_emergency": False,
                "priority": "UNCLASSIFIED",
                "note": "V2 modelleri henüz eğitilmedi, scripts/train_v2.py ile eğitebilirsiniz."
            }

        # 1. Kademe: İnsan Varlığı Tespiti (34-boyutlu V2 vektörü)
        f_human = human_features_v2(y, sr=sr)
        human_probs = self.human_model.predict_proba(f_human.reshape(1, -1))[0]
        classes_h = list(self.human_model.classes_)
        h_idx = classes_h.index("human") if "human" in classes_h else 1
        human_prob = float(human_probs[h_idx])

        if human_prob < self.human_threshold:
            return {
                "status": "no_human",
                "human_prob": human_prob,
                "rms": rms_val,
                "is_emergency": False,
                "priority": "NONE"
            }

        # 2. Kademe: 5 Sınıflı Acil Durum & Travma Tespiti (20-boyutlu V2 vektörü)
        f_emerg = emergency_features_v2(y, sr=sr)
        emerg_probs = self.emergency_model.predict_proba(f_emerg.reshape(1, -1))[0]
        classes_e = list(self.emergency_model.classes_)

        best_idx = int(np.argmax(emerg_probs))
        predicted_state = str(classes_e[best_idx])
        state_confidence = float(emerg_probs[best_idx])

        # Acil durum olasılığı: normal sınıfı dışındakilerin toplamı
        normal_idx = classes_e.index("normal") if "normal" in classes_e else -1
        p_normal = float(emerg_probs[normal_idx]) if normal_idx >= 0 else 0.0
        emergency_prob = float(1.0 - p_normal)

        is_emergency = (predicted_state != "normal") and (emergency_prob >= EMERGENCY_PROB_THRESHOLD)
        priority = ALERT_PRIORITY.get(predicted_state, "NORMAL_SPEECH") if is_emergency else "NORMAL_SPEECH"

        all_probs = {str(c): float(p) for c, p in zip(classes_e, emerg_probs)}

        return {
            "status": "DETECTED",
            "state": predicted_state,
            "priority": priority,
            "is_emergency": is_emergency,
            "human_prob": human_prob,
            "state_confidence": state_confidence,
            "emergency_prob": emergency_prob,
            "rms": rms_val,
            "all_probs": all_probs
        }

    def analyze_file(self, path):
        y, orig_sr = librosa.load(path, sr=SR)
        return self.analyze_audio_array(y, sr=orig_sr)


# Global tekil nesne
_default_pipeline_v2 = None


def get_pipeline_v2():
    global _default_pipeline_v2
    if _default_pipeline_v2 is None:
        _default_pipeline_v2 = PipelineV2()
    return _default_pipeline_v2


def analyze_audio_v2(path_or_array, sr=SR):
    pipeline = get_pipeline_v2()
    if isinstance(path_or_array, str):
        return pipeline.analyze_file(path_or_array)
    return pipeline.analyze_audio_array(path_or_array, sr=sr)
