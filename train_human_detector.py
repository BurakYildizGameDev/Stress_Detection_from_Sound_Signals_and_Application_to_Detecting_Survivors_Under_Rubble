import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ===============================
# FEATURE EXTRACTION
# ===============================

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=4)

        # RMS
        rms = np.mean(librosa.feature.rms(y=y))

        # ZCR
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))

        # Spectral
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        return np.hstack([
            rms, zcr, centroid, rolloff,
            mfcc_mean, mfcc_std
        ])

    except:
        return None

# ===============================
# DATA LOADER
# ===============================

def load_dataset():
    X, y = [], []

    human_dirs = [
        "data/human/berlin",
        "data/human/ravdess",
        "data/human/tess",
        "data/human/savee",
        "data/human/jl_corpus",
        "data/human/subesco",
    ]

    non_human_dirs = [
        "data_augmented_non_human/esc50",
        "data_augmented_non_human/urban",
    ]

    print("\nHuman sesler yükleniyor...")
    for d in human_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".wav") or f.endswith(".mp3"):
                    feat = extract_features(os.path.join(root, f))
                    if feat is not None:
                        X.append(feat)
                        y.append(1)  # human

    print("Non-human sesler yükleniyor...")
    for d in non_human_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".wav") or f.endswith(".mp3"):
                    feat = extract_features(os.path.join(root, f))
                    if feat is not None:
                        X.append(feat)
                        y.append(0)  # non-human

    return np.array(X), np.array(y)

# ===============================
# TRAIN
# ===============================

def train():
    print("\nHUMAN PRESENCE DETECTOR EĞİTİMİ\n")

    X, y = load_dataset()

    print(f"Toplam örnek: {len(X)}")
    print(f"Human: {np.sum(y==1)} | Non-human: {np.sum(y==0)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\n🎯 Accuracy: %{acc*100:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Non-Human", "Human"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/human_detector_rf.pkl")
    joblib.dump(scaler, "models/human_detector_scaler.pkl")

    print("\n✅ Model kaydedildi")

if __name__ == "__main__":
    train()
