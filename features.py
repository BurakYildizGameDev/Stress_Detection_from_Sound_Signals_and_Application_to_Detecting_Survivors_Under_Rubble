"""
Öznitelik çıkarımı için tek kaynak.

Hem eğitim scriptleri (extract_*.py) hem de canlı pipeline bu modülü kullanır.
Böylece modelin eğitimde gördüğü öznitelikler ile canlıda verilenler
yapısal olarak aynı kalır.
"""
import numpy as np
import librosa

HUMAN_SR = 22050
EMERGENCY_SR = 16000
N_MFCC = 13

HUMAN_N_FEATURES = 2 * N_MFCC + 2      # 28
EMERGENCY_N_FEATURES = N_MFCC + 2      # 15

# Öznitelik tanımının kimliği. Aşağıdaki fonksiyonlardan biri değişirse
# ilgili sürüm artırılmalı; pipeline, model metadata'sındaki spec ile bunu
# karşılaştırır ve uyuşmayan modeli yüklemez. clip_sec: eğitimde kaydın ilk
# kaç saniyesinin kullanıldığı (None = tamamı).
FEATURE_SPECS = {
    "human": {"version": 1, "sr": HUMAN_SR, "n_features": HUMAN_N_FEATURES,
              "clip_sec": 2.0,
              "desc": "13 MFCC mean + 13 MFCC std + ZCR + RMS"},
    "emergency": {"version": 1, "sr": EMERGENCY_SR, "n_features": EMERGENCY_N_FEATURES,
                  "clip_sec": None,
                  "desc": "13 MFCC mean + RMS + spectral centroid"},
}


def _resample(y, sr, target_sr):
    if sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)


def human_features(y, sr=HUMAN_SR):
    """
    İnsan sesi tespiti için 28 öznitelik (22.05 kHz):
    - 13 MFCC ortalama
    - 13 MFCC standart sapma
    - 1 ZCR ortalama
    - 1 RMS ortalama
    """
    y = _resample(y, sr, HUMAN_SR)
    mfcc = librosa.feature.mfcc(y=y, sr=HUMAN_SR, n_mfcc=N_MFCC)
    return np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.rms(y=y)),
    ]).astype(np.float32)


def emergency_features(y, sr=EMERGENCY_SR):
    """
    Acil durum sınıflandırması için 15 öznitelik (16 kHz):
    - 13 MFCC ortalama
    - 1 RMS ortalama (index 13)
    - 1 Spectral Centroid ortalama (index 14)
    """
    y = _resample(y, sr, EMERGENCY_SR)
    return np.hstack([
        np.mean(librosa.feature.mfcc(y=y, sr=EMERGENCY_SR, n_mfcc=N_MFCC), axis=1),
        np.mean(librosa.feature.rms(y=y)),
        np.mean(librosa.feature.spectral_centroid(y=y, sr=EMERGENCY_SR)),
    ]).astype(np.float32)


FEATURE_FUNCS = {"human": human_features, "emergency": emergency_features}
