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


# Operatörün taşıyabileceği yanlış alarm oranı (saha kararı): %20'de gerçek
# sözsüz vokalizasyonların ~%70'i yakalanır, %10'da yalnızca ~%40'ı.
TARGET_FALSE_ALARM = 0.20


def validation_split(d, train, val_fraction=0.15, seed=42):
    """Eğitim gruplarının (konuşmacı / kaynak) bir kısmını doğrulamaya ayırır.
    Dönüş: (fit, is_val) maskeleri; ikisi de yalnızca eğitim satırlarını kapsar."""
    groups = sorted(set(d["group"][train]))
    rng = np.random.default_rng(seed)
    val_groups = set(rng.choice(groups, size=max(1, int(len(groups) * val_fraction)), replace=False))
    is_val = np.isin(d["group"], sorted(val_groups)) & train
    return train & ~is_val, is_val


def pick_human_threshold(p_human, y, target_fa=TARGET_FALSE_ALARM):
    """Insan dışı kliplerde yanlış alarmı target_fa altında tutan en düşük eşik
    (en düşük eşik = en az kaçırma). Bulunamazsa 0.5."""
    neg, pos = y == "non_human", y == "human"
    for t in np.round(np.arange(0.05, 1.0, 0.01), 2):
        if np.mean(p_human[neg] >= t) <= target_fa:
            return float(t)
    return 0.5


def choose_human_threshold(d, train, target_fa=TARGET_FALSE_ALARM, val_fraction=0.15, seed=42,
                           make_model=lambda: make_model_v2("human"), verbose=True):
    """İnsan-sesi eşiğini test verisine bakmadan seçer: eğitim gruplarının bir
    kısmı doğrulamaya ayrılır, model onlarsız eğitilir ve doğrulamadaki insan dışı
    orijinal kliplerde yanlış alarm oranını target_fa altında tutan en düşük eşik
    seçilir (en düşük eşik = en az kaçırma)."""
    fit, is_val = validation_split(d, train, val_fraction, seed)
    model = make_model().fit(d["X"][fit], d["y"][fit])
    val = is_val & (d["transform"] == "original") & (d["synth"] == "none")
    p_human = model.predict_proba(d["X"][val])[:, list(model.classes_).index("human")]
    y = d["y"][val]
    t = pick_human_threshold(p_human, y, target_fa)
    if verbose:
        neg, pos = y == "non_human", y == "human"
        print(f"eşik {t:.2f}: doğrulamada yanlış alarm {np.mean(p_human[neg] >= t):.3f}, "
              f"insan recall {np.mean(p_human[pos] >= t):.3f} (n_neg={neg.sum()}, n_pos={pos.sum()})")
    return t


def threshold_table(d, model, threshold):
    """Seçilen eşikte test sonuçları (argmax yerine olasılık eşiği)."""
    p_human = model.predict_proba(d["X"])[:, list(model.classes_).index("human")]
    pred = np.where(p_human >= threshold, "human", "non_human")
    test = (d["split"] == "test") & (d["synth"] == "none")
    out = {}
    for cond in dict.fromkeys(d["condition"][test].tolist()):
        at = test & (d["condition"] == cond)
        out[cond] = {ds: {"n": int((at & (d["dataset"] == ds)).sum()),
                          "accuracy": float(np.mean(pred[at & (d["dataset"] == ds)]
                                                    == d["y"][at & (d["dataset"] == ds)]))}
                     for ds in sorted(set(d["dataset"][at]))}
    return out


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


def train_task_v2(task, class_weight="cost", save=True, target_fa=TARGET_FALSE_ALARM):
    d = load_features_v2(task)
    train = d["split"] == "train"
    print(f"\n=== {task}_v2 (sınıf ağırlığı: {class_weight if task == 'emergency' else 'balanced'}) ===")
    print(f"train: {train.sum()} satır ({len(set(d['group'][train]))} konuşmacı/grup), "
          f"test: {(~train).sum()} satır ({len(set(d['group'][~train]))} konuşmacı/grup)")
    print("train sınıfları:", dict(Counter(d["y"][train].tolist())))

    threshold = choose_human_threshold(d, train, target_fa) if task == "human" else None
    model = make_model_v2(task, class_weight).fit(d["X"][train], d["y"][train])
    report = evaluate(task, model, d)
    print_report(task, report)
    if threshold is not None:
        report["threshold"] = {"value": threshold, "target_false_alarm": target_fa,
                               "selected_on": "15% of train groups held out (not test)",
                               "test_accuracy_by_dataset": threshold_table(d, model, threshold)}
        print(f"\nSeçilen eşik {threshold:.2f} ile test doğruluğu (veri seti bazında):")
        for cond, per_ds in report["threshold"]["test_accuracy_by_dataset"].items():
            print(f"  {cond:24s} " + "  ".join(f"{ds} {r['accuracy']:.2f}" for ds, r in per_ds.items()))

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
        "human_threshold": threshold,
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
    p.add_argument("--target-false-alarm", type=float, default=TARGET_FALSE_ALARM,
                   help="insan-sesi eşiği seçilirken doğrulamada izin verilen yanlış alarm oranı")
    args = p.parse_args()
    for t in TASKS if args.task == "all" else (args.task,):
        train_task_v2(t, args.class_weight, save=not args.no_save,
                      target_fa=args.target_false_alarm)


if __name__ == "__main__":
    main()
