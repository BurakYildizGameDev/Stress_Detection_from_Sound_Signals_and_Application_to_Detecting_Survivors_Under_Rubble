"""
train_v2.py - v2 modellerini eğitir, görülmemiş konuşmacılar ve gerçek dış veri
üzerinde değerlendirir; models/ ve reports/ altına yazar.

Değerlendirme (hepsi test bölmesi, eğitimde hiç görülmemiş konuşmacılar):
  corpus      Korpus test kayıtları + onlardan sentezlenen fısıltı/inleme,
              her enkaz koşulunda (clean, rubble_physical_mild/medium/severe).
  by_synth    Aynı satırlar, kaynağa göre ayrık: gerçek kayıtlar / sentetik
              fısıltı / sentetik inleme. Sentetik sonuçlar dönüştürücünün
              kendisini de ölçer, gerçek fısıltı/inleme başarısı yerine geçmez.
  external    eval_only veri setleri (VIVAE gerçek ağrı/korku vokalizasyonları,
              field/ kayıtları); modelin hiç görmediği gerçek veri.

Sınıf ağırlıkları:
  cost      normal 1, stress 3, panic 3, moan 6, whisper 8 (kaçırmanın bedeli)
  balanced  sklearn "balanced" (karşılaştırma için)

Kullanım:
    python scripts/train_v2.py --task all
    python scripts/train_v2.py --task emergency --class-weight balanced --no-save
"""
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import EMERGENCY_CLASSES_V2, EVAL_ONLY_DATASETS, FEATURES_DIR, MODELS_DIR, ROOT
from evaluation import format_summary, summarize
from features_v2 import FEATURE_SPECS_V2
from pipeline_v2 import MODEL_FILES_V2

TASKS = ("emergency", "human")
REPORTS_DIR = os.path.join(ROOT, "reports")
LABELS = {"emergency": list(EMERGENCY_CLASSES_V2), "human": ["human", "non_human"]}

# Maliyet duyarlı sınıf ağırlıkları (fısıltı ve inleme gözden kaçamaz)
COST_WEIGHTS_V2 = {
    "normal": 1.0,
    "stress": 3.0,
    "panic": 3.0,
    "moan": 6.0,
    "whisper": 8.0,
}


def load_features_v2(task):
    path = os.path.join(FEATURES_DIR, f"{task}_v2.npz")
    if not os.path.exists(path):
        sys.exit(f"Öznitelik dosyası yok: {os.path.relpath(path, ROOT)}\n"
                 f"Önce: python scripts/extract_features_v2.py --task {task}")
    with np.load(path) as d:
        if "condition" not in d.files:
            sys.exit("Eski biçimli öznitelik dosyası; extract_features_v2.py'yi yeniden çalıştırın.")
        return {k: d[k] for k in d.files}


def make_model_v2(task, class_weight="cost"):
    if task == "human":
        return RandomForestClassifier(n_estimators=250, max_depth=20, min_samples_leaf=2,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    weights = COST_WEIGHTS_V2 if class_weight == "cost" else "balanced"
    return RandomForestClassifier(n_estimators=350, max_depth=22, min_samples_leaf=2,
                                  class_weight=weights, random_state=42, n_jobs=-1)


def prediction_counts(y_true, y_pred):
    """Tek sınıflı dış veri için: her gerçek sınıf hangi sınıflara tahmin edildi."""
    return {str(t): dict(Counter(y_pred[y_true == t].tolist())) for t in sorted(set(y_true))}


def evaluate(task, model, d):
    labels = LABELS[task]
    test = d["split"] == "test"
    external = np.isin(d["dataset"], sorted(EVAL_ONLY_DATASETS))
    pred = model.predict(d["X"])
    conditions = [c for c in dict.fromkeys(d["condition"][test].tolist())]

    def summ(mask):
        return summarize(task, d["y"][mask], pred[mask], labels)

    report = {"corpus": {}, "by_synth": {}, "by_dataset": {}, "external": {}}
    for cond in conditions:
        at = test & (d["condition"] == cond)
        report["corpus"][cond] = summ(at & ~external)
        report["by_synth"][cond] = {}
        for synth in sorted(set(d["synth"][at & ~external])):
            m = at & ~external & (d["synth"] == synth)
            report["by_synth"][cond][synth] = {
                "n": int(m.sum()),
                "accuracy": float(np.mean(pred[m] == d["y"][m])),
                "predictions": prediction_counts(d["y"][m], pred[m]),
            }
        report["by_dataset"][cond] = {}
        for ds in sorted(set(d["dataset"][at & ~external])):
            m = at & ~external & (d["dataset"] == ds) & (d["synth"] == "none")
            report["by_dataset"][cond][ds] = {
                "n": int(m.sum()),
                "accuracy": float(np.mean(pred[m] == d["y"][m])) if m.any() else float("nan"),
                "predictions": prediction_counts(d["y"][m], pred[m]),
            }
        for ds in sorted(set(d["dataset"][at & external])):
            m = at & (d["dataset"] == ds)
            report["external"].setdefault(ds, {})[cond] = {
                "n": int(m.sum()),
                "accuracy": float(np.mean(pred[m] == d["y"][m])),
                "predictions": prediction_counts(d["y"][m], pred[m]),
            }
    return report


def print_report(task, report):
    clean = report["corpus"].get("clean")
    if clean:
        print("\nGörülmemiş konuşmacılar, temiz koşul (gerçek + sentetik):\n" + format_summary(clean))
    print("\nEnkaz koşullarına göre (korpus testi):")
    for cond, s in report["corpus"].items():
        rec = "  ".join(f"{lab} {c['recall']:.2f}" for lab, c in s["per_class"].items())
        print(f"  {cond:24s} macro-F1 {s['macro_f1']:.3f} | recall: {rec}")
    print("\nVeri setine göre doğruluk (gerçek kayıtlar, clean / severe):")
    for ds, r in report["by_dataset"].get("clean", {}).items():
        sev = report["by_dataset"].get("rubble_physical_severe", {}).get(ds, {})
        print(f"  {ds:12s} n={r['n']:4d}  {r['accuracy']:.3f} / {sev.get('accuracy', float('nan')):.3f}")
    for ds, per_cond in report["external"].items():
        print(f"\nDış gerçek veri: {ds}")
        for cond, r in per_cond.items():
            print(f"  {cond:24s} n={r['n']:4d} doğruluk {r['accuracy']:.3f}  {r['predictions']}")


def train_task_v2(task, class_weight="cost", save=True):
    d = load_features_v2(task)
    train = d["split"] == "train"
    print(f"\n=== {task}_v2 (sınıf ağırlığı: {class_weight if task == 'emergency' else 'balanced'}) ===")
    print(f"train: {train.sum()} satır ({len(set(d['group'][train]))} konuşmacı/grup), "
          f"test: {(~train).sum()} satır ({len(set(d['group'][~train]))} konuşmacı/grup)")
    print("train sınıfları:", dict(Counter(d["y"][train].tolist())))

    model = make_model_v2(task, class_weight).fit(d["X"][train], d["y"][train])
    report = evaluate(task, model, d)
    print_report(task, report)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    suffix = "" if class_weight == "cost" or task == "human" else f"_{class_weight}"
    report_path = os.path.join(REPORTS_DIR, f"{task}_v2{suffix}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n-> {os.path.relpath(report_path, ROOT)}")
    if not save:
        return report

    with open(os.path.join(FEATURES_DIR, f"{task}_v2.json"), encoding="utf-8") as f:
        feat_meta = json.load(f)
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, MODEL_FILES_V2[task])
    joblib.dump(model, model_path, compress=3)
    meta = {
        "task": f"{task}_v2",
        "labels": [str(c) for c in model.classes_],
        "feature_spec": FEATURE_SPECS_V2[task],
        "sklearn_version": sklearn.__version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": feat_meta["manifest_sha256"],
        "class_weight": COST_WEIGHTS_V2 if task == "emergency" and class_weight == "cost" else "balanced",
        "n_train_rows": int(train.sum()),
        "eval_protocol": "held-out speakers; synthetic whisper/moan from test speakers; "
                         "fixed rubble conditions; eval_only datasets as external real data",
        "metrics_clean": report["corpus"].get("clean"),
        "report": os.path.relpath(report_path, ROOT).replace(os.sep, "/"),
    }
    with open(os.path.splitext(model_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"-> {os.path.relpath(model_path, ROOT)} (+ .json)")
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS + ("all",), default="all")
    p.add_argument("--class-weight", choices=("cost", "balanced"), default="cost")
    p.add_argument("--no-save", action="store_true", help="modeli kaydetme, yalnızca raporla")
    args = p.parse_args()
    for t in TASKS if args.task == "all" else (args.task,):
        train_task_v2(t, args.class_weight, save=not args.no_save)


if __name__ == "__main__":
    main()
