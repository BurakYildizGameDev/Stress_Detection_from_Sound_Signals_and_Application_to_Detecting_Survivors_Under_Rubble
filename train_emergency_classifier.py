import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, os

X = np.load("features/X_emergency.npy")
y = np.load("features/y_emergency.npy")

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(Xtr, ytr)
print(classification_report(yte, model.predict(Xte)))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/emergency_classifier.pkl")
print("✅ MODEL KAYDEDİLDİ")
