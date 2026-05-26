import os
import numpy as np
import librosa
from tqdm import tqdm

BASE = "data_emergency"
LABELS = {"normal":0, "stress":1, "scream":2, "panic":3}

X, y = [], []

def feats(y_, sr):
    return np.hstack([
        np.mean(librosa.feature.mfcc(y=y_, sr=sr, n_mfcc=13), axis=1),
        np.mean(librosa.feature.rms(y=y_), axis=1),
        np.mean(librosa.feature.spectral_centroid(y=y_, sr=sr), axis=1)
    ])

for cls, lbl in LABELS.items():
    folder = os.path.join(BASE, cls)
    for f in tqdm(os.listdir(folder), desc=cls):
        if not f.endswith(".wav"): continue
        y_, sr = librosa.load(os.path.join(folder, f), sr=16000)
        X.append(feats(y_, sr))
        y.append(lbl)

os.makedirs("features", exist_ok=True)
np.save("features/X_emergency.npy", np.array(X))
np.save("features/y_emergency.npy", np.array(y))

print("✅ FEATURES OK")
