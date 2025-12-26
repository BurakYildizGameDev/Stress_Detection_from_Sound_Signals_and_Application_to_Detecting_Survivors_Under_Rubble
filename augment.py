import numpy as np
import librosa

def add_noise_snr(y, snr_db, rng):
    power = np.mean(y ** 2)
    if power == 0:
        return y.copy()
    noise = rng.standard_normal(len(y)) * np.sqrt(power / 10 ** (snr_db / 10))
    return (y + noise).astype(np.float32)

from scipy.signal import butter, sosfilt
def lowpass(y, sr, cutoff_hz, order=4):
    sos = butter(order, cutoff_hz, btype='low', fs=sr, output='sos')
    return sosfilt(sos, y).astype(np.float32)

def gain_db(y, db):
    return (y * 10 ** (db / 20)).astype(np.float32)

def pitch_shift(y, sr, steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps).astype(np.float32)
