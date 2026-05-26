import os
import numpy as np
import librosa
from tqdm import tqdm

HUMAN_DIR = "data_final/human"
NON_HUMAN_DIR = "data_final/non_human"
FEATURE_DIR = "features"

os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=3)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(zcr),
        np.mean(rms),
        np.mean(centroid)
    ])

    return features


def process_recursive(folder, label):
    X, y = [], []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                try:
                    feat = extract_features(path)
                    X.append(feat)
                    y.append(label)
                except Exception as e:
                    print(f"Hata atlandı: {path}")

    return X, y


print("\nHuman sesler yükleniyor...")
X_h, y_h = process_recursive(HUMAN_DIR, 1)

print("Non-human sesler yükleniyor...")
X_nh, y_nh = process_recursive(NON_HUMAN_DIR, 0)

X = np.array(X_h + X_nh)
y = np.array(y_h + y_nh)

np.save(os.path.join(FEATURE_DIR, "X_human_detector.npy"), X)
np.save(os.path.join(FEATURE_DIR, "y_human_detector.npy"), y)

print("\n✅ Feature extraction tamamlandı")
print(f"Toplam örnek: {len(X)}")
print(f"Human: {sum(y)} | Non-human: {len(y) - sum(y)}")
