import numpy as np
import librosa

HUMAN_SR = 22050
EMERGENCY_SR = 16000
N_MFCC = 13

def _resample(y, sr, target_sr):
    if sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)

def extract_rms(y):
    return np.mean(librosa.feature.rms(y=y))

def extract_zcr(y):
    return np.mean(librosa.feature.zero_crossing_rate(y))

def extract_centroid(y, sr):
    return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

def extract_mfcc(y, sr, n_mfcc=N_MFCC):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

def mfcc_stats(mfcc):
    return np.mean(mfcc, axis=1), np.std(mfcc, axis=1)

HUMAN_N_FEATURES = 2 * N_MFCC + 2
EMERGENCY_N_FEATURES = N_MFCC + 2
FEATURE_SPECS = {
    'human': {'version': 1, 'sr': HUMAN_SR, 'n_features': HUMAN_N_FEATURES, 'clip_sec': 2.0},
    'emergency': {'version': 1, 'sr': EMERGENCY_SR, 'n_features': EMERGENCY_N_FEATURES, 'clip_sec': None},
}

def human_features(y, sr=HUMAN_SR):
    y = _resample(y, sr, HUMAN_SR)
    mfcc = librosa.feature.mfcc(y=y, sr=HUMAN_SR, n_mfcc=N_MFCC)
    return np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(librosa.feature.zero_crossing_rate(y)),
        np.mean(librosa.feature.rms(y=y)),
    ]).astype(np.float32)
