import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

X = np.load("features_mid/X.npy")
y = np.load("features_mid/y.npy")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_leaf=5,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n📊 RAPOR")
print(classification_report(y_test, y_pred, target_names=["NON-HUMAN", "HUMAN"]))

joblib.dump(model, "human_detector_mid.pkl")
print("✅ MODEL KAYDEDİLDİ")
