import os
import numpy as np
import librosa
from tqdm import tqdm

DATA_DIR = "data_mid"
OUT_DIR = "features_mid"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_features(path):
    y, sr = librosa.load(path, sr=22050, duration=3.0)

    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = [
        rms.mean(), rms.std(),
        zcr.mean(), zcr.std(),
        centroid.mean(), centroid.std()
    ]

    for m in mfcc:
        features.append(m.mean())
        features.append(m.var())

    return np.array(features)

def process(label, y_value):
    X, y = [], []

    base = os.path.join(DATA_DIR, label)
    for root, _, files in os.walk(base):
        for f in tqdm(files, desc=label):
            if f.endswith(".wav"):
                try:
                    feat = extract_features(os.path.join(root, f))
                    X.append(feat)
                    y.append(y_value)
                except:
                    pass

    return np.array(X), np.array(y)

if __name__ == "__main__":
    Xh, yh = process("human", 1)
    Xn, yn = process("non_human", 0)

    X = np.vstack([Xh, Xn])
    y = np.hstack([yh, yn])

    np.save(f"{OUT_DIR}/X.npy", X)
    np.save(f"{OUT_DIR}/y.npy", y)

    print("✅ FEATURES KAYDEDİLDİ:", X.shape)
