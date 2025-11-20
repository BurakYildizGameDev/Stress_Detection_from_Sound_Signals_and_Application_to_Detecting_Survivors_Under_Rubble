import numpy as np
import librosa

HUMAN_SR = 22050
EMERGENCY_SR = 16000
N_MFCC = 13

def _resample(y, sr, target_sr):
    if sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr)
