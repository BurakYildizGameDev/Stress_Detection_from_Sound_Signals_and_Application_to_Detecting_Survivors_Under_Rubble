"""
Eğitilmiş modelleri üç protokolle değerlendirir ve sonucu reports/ altına yazar.

  held-out      Kayıtlı modeli görülmemiş konuşmacılar (test bölmesi) üzerinde ölçer.
  cross-dataset Her veri setini sırayla dışarıda bırakıp kalanlarla eğitir, dışarıda
                bırakılan veri setinde ölçer (dil / kayıt ortamı değişince ne olur).
  robustness    Test kayıtlarını gürültü, zayıflama ve alçak geçiren filtreden
                geçirip kayıtlı modeli her koşulda ölçer (bkz. augment.CONDITIONS).

Kullanım:
    python scripts/evaluate.py --task emergency
    python scripts/evaluate.py --task emergency --protocol cross-dataset
    python scripts/evaluate.py --task all --protocol all
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
from joblib import Parallel, delayed

from augment import CONDITIONS
from dataset import ROOT
from evaluation import (NEGATIVE_CLASS, format_summary, held_out_mask,
                        load_features, make_model, summarize)
from features import FEATURE_FUNCS, FEATURE_SPECS
from pipeline import load_model

TASKS = ("emergency", "human")
PROTOCOLS = ("held-out", "cross-dataset", "robustness")
REPORTS_DIR = os.path.join(ROOT, "reports")


def held_out(task):
    d = load_features(task)
    m = held_out_mask(d)
    loaded = load_model(task)
    pred = np.array(loaded.labels)[np.argmax(loaded.model.predict_proba(d["X"][m]), axis=1)]
    s = summarize(task, d["y"][m], pred, sorted(set(d["y"]) | set(pred)))
    print(format_summary(s))
    return s


def cross_dataset(task):
    """Her veri setini sırayla dışarıda bırakır. İnsan-sesi modelinde yalnızca
    insan veri setleri dışarıda bırakılır; negatif sınıfın test bölmesi (ESC-50
    fold 5) her turda teste eklenir ki yanlış alarm da ölçülebilsin."""
    d = load_features(task)
    labels = sorted(set(d["y"]))
    negative = NEGATIVE_CLASS[task]
    shared_test = (d["y"] == negative) & (d["split"] == "test") if task == "human"         else np.zeros(len(d["y"]), dtype=bool)
    results = {}
    for ds in sorted(set(d["dataset"])):
        held = d["dataset"] == ds
        if task == "human" and np.all(d["y"][held] == negative):
            continue
        train = ~held & ~shared_test
        test = (held | shared_test) & (d["transform"] == "original")
        if len(set(d["y"][train])) < len(labels):
            print(f"[{ds}] atlandı: kalan veride her sınıf yok")
            continue
        model = make_model(task).fit(d["X"][train], d["y"][train])
        s = summarize(task, d["y"][test], model.predict(d["X"][test]), labels)
        results[ds] = s
        print(f"[{ds:9s}] n={s['n']:5d}  macro-F1 {s['macro_f1']:.3f}  "
              f"yanlış alarm {s['false_alarm_rate']:.3f}  kaçırma {s['miss_rate']:.3f}")
    return results


def _condition_features(task, path, condition):
    spec = FEATURE_SPECS[task]
    sr = spec["sr"]
    y, _ = librosa.load(os.path.join(ROOT, path), sr=sr, duration=spec["clip_sec"])
    rng = np.random.default_rng(zlib.crc32(f"{path}|{condition}".encode()))
    return FEATURE_FUNCS[task](CONDITIONS[condition](y, sr, rng), sr)


def robustness(task, jobs):
    d = load_features(task)
    m = held_out_mask(d)
    paths, y_true = d["path"][m], d["y"][m]
    loaded = load_model(task)
    labels = sorted(set(d["y"]) | set(loaded.labels))
    results = {}
    for cond in CONDITIONS:
        X = np.array(Parallel(n_jobs=jobs)(
            delayed(_condition_features)(task, p, cond) for p in paths))
        pred = np.array(loaded.labels)[np.argmax(loaded.model.predict_proba(X), axis=1)]
        s = summarize(task, y_true, pred, labels)
        results[cond] = s
        print(f"[{cond:12s}] macro-F1 {s['macro_f1']:.3f}  "
              f"yanlış alarm {s['false_alarm_rate']:.3f}  kaçırma {s['miss_rate']:.3f}")
    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS + ("all",), required=True)
    ap.add_argument("--protocol", choices=PROTOCOLS + ("all",), default="held-out")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    for task in TASKS if args.task == "all" else (args.task,):
        for protocol in PROTOCOLS if args.protocol == "all" else (args.protocol,):
            print(f"\n=== {task} / {protocol} (negatif sınıf: {NEGATIVE_CLASS[task]}) ===")
            if protocol == "held-out":
                result = held_out(task)
            elif protocol == "cross-dataset":
                result = cross_dataset(task)
            else:
                result = robustness(task, args.jobs)
            out = os.path.join(REPORTS_DIR, f"{task}_{protocol}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"-> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
