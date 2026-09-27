"""
Eğitim ve değerlendirme scriptlerinin ortak parçaları: öznitelik dosyasını
yükleme, model tanımı ve metrikler.

Enkaz senaryosunda iki hata türü farklı bedeller taşır, bu yüzden ayrı raporlanır:
- Yanlış alarm oranı: negatif (normal / non_human) kaydın alarm sınıfına düşme oranı
- Kaçırma oranı: pozitif (stress+panic / human) kaydın negatif sanılma oranı
"""
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from dataset import FEATURES_DIR, ROOT

NEGATIVE_CLASS = {"emergency": "normal", "human": "non_human"}

MODEL_PARAMS = {
    "emergency": dict(n_estimators=300, max_depth=15, class_weight="balanced"),
    "human": dict(n_estimators=300, max_depth=20, min_samples_leaf=5, class_weight="balanced"),
}


def make_model(task, seed=42):
    return RandomForestClassifier(**MODEL_PARAMS[task], n_jobs=-1, random_state=seed)


def load_features(task):
    path = os.path.join(FEATURES_DIR, f"{task}.npz")
    if not os.path.exists(path):
        raise SystemExit(
            f"Öznitelik dosyası yok: {os.path.relpath(path, ROOT)}\n"
            f"Önce çalıştırın: python scripts/extract_features.py --task {task}")
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def held_out_mask(data):
    """Test bölmesindeki orijinal kayıtlar (artırılmış kopya içermez)."""
    return (data["split"] == "test") & (data["transform"] == "original")


def binary_rates(y_true, y_pred, negative):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    neg = y_true == negative
    pos = ~neg
    false_alarm = float(np.mean(y_pred[neg] != negative)) if neg.any() else float("nan")
    miss = float(np.mean(y_pred[pos] == negative)) if pos.any() else float("nan")
    return false_alarm, miss


def summarize(task, y_true, y_pred, labels):
    labels = list(labels)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    fa, miss = binary_rates(y_true, y_pred, NEGATIVE_CLASS[task])
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "false_alarm_rate": fa,
        "miss_rate": miss,
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(s[i])}
                      for i, lab in enumerate(labels)},
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def format_summary(s):
    labels = s["labels"]
    w = max(10, *(len(l) for l in labels)) + 2
    lines = [f"{'':{w}}precision  recall    f1  support"]
    for lab in labels:
        c = s["per_class"][lab]
        lines.append(f"{lab:{w}}{c['precision']:9.3f}{c['recall']:8.3f}{c['f1']:6.3f}{c['support']:9d}")
    lines += [
        "",
        f"accuracy {s['accuracy']:.3f} | macro-F1 {s['macro_f1']:.3f} | n={s['n']}",
        f"yanlış alarm oranı {s['false_alarm_rate']:.3f} | kaçırma oranı {s['miss_rate']:.3f}",
        "",
        "confusion matrix (satır = gerçek, sütun = tahmin)",
        f"{'':{w}}" + "".join(f"{l[:9]:>10}" for l in labels),
    ]
    for lab, row in zip(labels, s["confusion_matrix"]):
        lines.append(f"{lab:{w}}" + "".join(f"{v:10d}" for v in row))
    return "\n".join(lines)
