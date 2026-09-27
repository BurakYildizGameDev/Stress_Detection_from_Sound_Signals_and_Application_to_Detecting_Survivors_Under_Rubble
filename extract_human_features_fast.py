import os
import numpy as np
import librosa
from tqdm import tqdm

from features import HUMAN_SR, human_features

def extract_features(path):
    y, sr = librosa.load(path, sr=HUMAN_SR, duration=2)
    return human_features(y, sr)

def process(folder, label):
    X, y = [], []
    files = []
    for root, _, fs in os.walk(folder):
        for f in fs:
            if f.endswith(".wav"):
                files.append(os.path.join(root, f))

    for f in tqdm(files, desc=folder):
        try:
            X.append(extract_features(f))
            y.append(label)
        except:
            pass

    return X, y

Xh, yh = process("data_sampled/human", 1)
Xn, yn = process("data_sampled/non_human", 0)

X = np.array(Xh + Xn)
y = np.array(yh + yn)

os.makedirs("features", exist_ok=True)
np.save("features/X_human_detector.npy", X)
np.save("features/y_human_detector.npy", y)

print("✅ FEATURES KAYDEDİLDİ")
