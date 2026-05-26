import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

print("\nHUMAN PRESENCE DETECTOR (MID MODEL)\n")

# Features yükle
X = np.load("features/X_human_detector.npy")
y = np.load("features/y_human_detector.npy")

print(f"Toplam örnek: {len(X)}")
print(f"Human: {sum(y==1)} | Non-human: {sum(y==0)}")

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

print("Model eğitiliyor...")
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)

print("\n📊 RAPOR")
print(classification_report(
    y_test,
    y_pred,
    target_names=["NON-HUMAN", "HUMAN"]
))

# Kaydet
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/human_detector_mid.pkl")

print("✅ MID MODEL KAYDEDİLDİ → models/human_detector_mid.pkl")
