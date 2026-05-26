import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

X = np.load("features/X_human_detector.npy")
y = np.load("features/y_human_detector.npy")

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    random_state=42
)

model.fit(Xtr, ytr)
yp = model.predict(Xte)

print("\n📊 RAPOR")
print(classification_report(yte, yp, target_names=["NON-HUMAN", "HUMAN"]))

joblib.dump(model, "models/human_detector.pkl")
print("✅ MODEL KAYDEDİLDİ")
