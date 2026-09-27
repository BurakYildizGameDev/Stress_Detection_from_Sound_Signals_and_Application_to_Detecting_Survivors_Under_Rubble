"""
Manifest'teki kayıtlardan öznitelik çıkarır: features/<task>.npz

Artırma (augmentation) yalnızca train bölmesindeki kayıtlara uygulanır, test
kayıtları orijinal haliyle kalır. Her satır hangi kayıttan ve hangi dönüşümle
geldiğini (path, transform) ve konuşmacı grubunu (group) taşır.

Kullanım:
    python scripts/extract_features.py --task emergency
    python scripts/extract_features.py --task human --max-per-dataset 300
    python scripts/extract_features.py --task all --no-augment
"""
import argparse
import json
import os
import random
import sys
import zlib
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
from joblib import Parallel, delayed
from tqdm import tqdm

from augment import AUGMENTATIONS
from dataset import (EVAL_ONLY_DATASETS, FEATURES_DIR, MANIFEST_PATH, ROOT, SPLIT_SEED,
                     file_sha256, read_manifest)
from features import FEATURE_FUNCS, FEATURE_SPECS

TASKS = ("emergency", "human")


def features_path(task):
    return os.path.join(FEATURES_DIR, f"{task}.npz")


def select_rows(task, records, max_per_dataset, seed=SPLIT_SEED):
    """Göreve giren kayıtlar ve etiketleri. v1 yalnızca dış doğrulama veri
    setlerini (EVAL_ONLY_DATASETS) kullanmaz."""
    records = [r for r in records if r.dataset not in EVAL_ONLY_DATASETS]
    if task == "emergency":
        return [(r, r.emergency_class) for r in records if r.emergency_class]

    rows = [(r, "non_human") for r in records if r.role == "non_human"]
    by_ds = {}
    for r in records:
        if r.role == "human":
            by_ds.setdefault(r.dataset, []).append(r)
    for ds, recs in sorted(by_ds.items()):
        if max_per_dataset and len(recs) > max_per_dataset:
            recs = random.Random(f"{seed}-{ds}").sample(recs, max_per_dataset)
        rows += [(r, "human") for r in recs]
    return rows


def process(task, rec, augment):
    """Bir kayıt -> [(öznitelik, dönüşüm adı), ...]. Hata olursa (None, mesaj)."""
    spec = FEATURE_SPECS[task]
    sr = spec["sr"]
    func = FEATURE_FUNCS[task]
    try:
        y, _ = librosa.load(os.path.join(ROOT, rec.path), sr=sr, duration=spec["clip_sec"])
        out = [(func(y, sr), "original")]
        if augment and rec.split == "train":
            for name, fn in AUGMENTATIONS[rec.role].items():
                rng = np.random.default_rng(zlib.crc32(f"{rec.path}|{name}".encode()))
                out.append((func(fn(y, sr, rng), sr), name))
        return out, None
    except Exception as e:
        return None, f"{rec.path}: {e}"


def extract(task, augment=True, max_per_dataset=300, jobs=-1):
    records = read_manifest()
    rows = select_rows(task, records, max_per_dataset)
    labels = {label for _, label in rows}
    if len(labels) < 2:
        sys.exit(f"[{task}] manifest'te yeterli sınıf yok ({sorted(labels)}). "
                 "Eksik veri setlerini indirip manifest'i yeniden oluşturun.")

    results = Parallel(n_jobs=jobs, return_as="generator")(
        delayed(process)(task, rec, augment) for rec, _ in rows)

    cols = {k: [] for k in ("X", "y", "split", "group", "dataset", "path", "transform")}
    errors = []
    for (rec, label), (out, err) in tqdm(zip(rows, results), total=len(rows), desc=task):
        if err:
            errors.append(err)
            continue
        for feat, transform in out:
            cols["X"].append(feat)
            cols["y"].append(label)
            cols["split"].append(rec.split)
            cols["group"].append(rec.group)
            cols["dataset"].append(rec.dataset)
            cols["path"].append(rec.path)
            cols["transform"].append(transform)

    if errors:
        print(f"\n{len(errors)} kayıt okunamadı, örnek:")
        for e in errors[:5]:
            print("  " + e)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_path = features_path(task)
    np.savez_compressed(out_path, X=np.array(cols.pop("X"), dtype=np.float32),
                        **{k: np.array(v) for k, v in cols.items()})

    meta = {
        "task": task,
        "feature_spec": FEATURE_SPECS[task],
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "augmentations": {role: list(a) for role, a in AUGMENTATIONS.items()} if augment else {},
        "max_per_dataset": max_per_dataset if task == "human" else None,
        "n_rows": len(cols["y"]),
        "n_failed": len(errors),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(out_path[:-4] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[{task}] {meta['n_rows']} satır -> {os.path.relpath(out_path, ROOT)}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS + ("all",), required=True)
    ap.add_argument("--no-augment", action="store_true", help="artırma yapma")
    ap.add_argument("--max-per-dataset", type=int, default=300,
                    help="insan-sesi modeli için veri seti başına en fazla kayıt (0 = sınırsız)")
    ap.add_argument("--jobs", type=int, default=-1, help="paralel işlem sayısı")
    args = ap.parse_args()

    for task in TASKS if args.task == "all" else (args.task,):
        extract(task, not args.no_augment, args.max_per_dataset, args.jobs)


if __name__ == "__main__":
    main()
