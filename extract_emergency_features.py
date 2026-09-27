import os
import numpy as np
import librosa
from tqdm import tqdm

from features import EMERGENCY_SR, emergency_features

BASE = "data_emergency"
LABELS = {"normal":0, "stress":1, "scream":2, "panic":3}

X, y = [], []

for cls, lbl in LABELS.items():
    folder = os.path.join(BASE, cls)
    for f in tqdm(os.listdir(folder), desc=cls):
        if not f.endswith(".wav"): continue
        y_, sr = librosa.load(os.path.join(folder, f), sr=EMERGENCY_SR)
        X.append(emergency_features(y_, sr))
        y.append(lbl)

os.makedirs("features", exist_ok=True)
np.save("features/X_emergency.npy", np.array(X))
np.save("features/y_emergency.npy", np.array(y))

print("✅ FEATURES OK")
