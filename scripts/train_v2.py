"""
train_v2.py - 5 Sınıflı ve Zayıf Vokal Destekli Model Eğitim Scripti.

Bu script:
1. features/<task>_v2.npz öznitelik dosyalarını yükler.
2. Görülmemiş konuşmacılar (held-out test split) üzerinde konuşmacı bağımsız doğrulamayı korur.
3. Maliyete Duyarlı Kayıp (Cost-Sensitive Matrix / class_weight) uygulayarak fısıltı ve inleme
   gibi hayati sinyallerin kaçırılmasını engeller:
     - normal:  1.0
     - stress:  3.0
     - panic:   3.0
     - moan:    6.0  (Ağır travma/inleme)
     - whisper: 8.0  (Kritik evre fısıltı - en yüksek ceza)
4. Modeli ve eşlik eden .json meta verisini models/ dizinine kaydeder.

Kullanım:
    python scripts/train_v2.py --task emergency
    python scripts/train_v2.py --task human
    python scripts/train_v2.py --task all
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FEATURES_DIR, MODELS_DIR, ROOT
from features_v2 import FEATURE_SPECS_V2

TASKS = ("emergency", "human")

MODEL_FILES_V2 = {
    "human": "human_detector_v2.pkl",
    "emergency": "emergency_classifier_v2.pkl",
}

# Maliyet duyarlı sınıf ağırlıkları (fısıltı ve inleme gözden kaçamaz!)
COST_WEIGHTS_V2 = {
    "normal": 1.0,
    "stress": 3.0,
    "panic": 3.0,
    "moan": 6.0,
    "whisper": 8.0,
}


def load_features_v2(task):
    npz_path = os.path.join(FEATURES_DIR, f"{task}_v2.npz")
    if not os.path.exists(npz_path):
        sys.exit(
            f"[!] HATA: Öznitelik matrisi bulunamadı: {npz_path}\n"
            f"    Önce öznitelikleri çıkarın: python scripts/extract_features_v2.py --task {task}"
        )
    return np.load(npz_path, allow_pickle=True)


def make_model_v2(task):
    if task == "human":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    # emergency v2: 5 sınıflı maliyet ağırlıklı model
    return RandomForestClassifier(
        n_estimators=350,
        max_depth=22,
        min_samples_leaf=2,
        class_weight=COST_WEIGHTS_V2,
        random_state=42,
        n_jobs=-1,
    )


def train_task_v2(task, n_trees=None, max_depth=None):
    d = load_features_v2(task)
    split = d["split"]
    transform = d["transform"]

    # Eğitim verisi: train split'indeki tüm kayıtlar (orijinal + artıramalar)
    train_mask = split == "train"
    # Test verisi: test split'indeki yalnızca orijinal (orijinal kayıtlar)
    test_mask = (split == "test") & (transform == "original")

    if not test_mask.any():
        # Sentetik Whisper/Moan testi için tüm test kayıtlarını kullan
        test_mask = split == "test"

    labels = sorted(list(set(d["y"])))
    print(f"\n=======================================================")
    print(f"[*] Görev: {task}_v2 Modeli Eğitiliyor")
    print(f"[*] Sınıflar ({len(labels)}): {labels}")
    print(f"[*] Eğitim Satırları: {train_mask.sum()} | Test Satırları: {test_mask.sum()}")
    print(f"=======================================================")

    model = make_model_v2(task)
    if n_trees:
        model.set_params(n_estimators=n_trees)
    if max_depth:
        model.set_params(max_depth=max_depth)

    X_train, y_train = d["X"][train_mask], d["y"][train_mask]
    X_test, y_test = d["X"][test_mask], d["y"][test_mask]

    print("[*] Random Forest eğitimi yapılıyor (n_jobs=-1)...")
    model.fit(X_train, y_train)

    # Değerlendirme
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n--- Test Kümesi Başarı Raporu ---")
    print(f"Doğruluk (Accuracy): %{acc * 100:.2f}")
    print(f"Macro F1-Score:     %{f1_macro * 100:.2f}")
    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # Modeli diske kaydet
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_filename = MODEL_FILES_V2[task]
    model_path = os.path.join(MODELS_DIR, model_filename)
    joblib.dump(model, model_path, compress=3)

    # Metadata JSON
    features_meta_path = os.path.join(FEATURES_DIR, f"{task}_v2.json")
    feat_meta = {}
    if os.path.exists(features_meta_path):
        with open(features_meta_path, encoding="utf-8") as f:
            feat_meta = json.load(f)

    meta = {
        "task": f"{task}_v2",
        "labels": [str(c) for c in model.classes_],
        "feature_spec": FEATURE_SPECS_V2[task],
        "sklearn_version": sklearn.__version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": feat_meta.get("manifest_sha256", ""),
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": int(test_mask.sum()),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "classes": labels,
    }

    meta_path = os.path.splitext(model_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[+] Model başarıyla kaydedildi -> {os.path.relpath(model_path, ROOT)}")
    print(f"[+] Metadata başarıyla kaydedildi -> {os.path.relpath(meta_path, ROOT)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="V2 Model Eğitimi (Fısıltı ve İnleme Destekli)")
    p.add_argument("--task", choices=TASKS + ("all",), default="all")
    p.add_argument("--n-trees", type=int, default=None)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args()

    tasks = TASKS if args.task == "all" else (args.task,)
    for t in tasks:
        train_task_v2(t, n_trees=args.n_trees, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
