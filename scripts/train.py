"""
Modeli train bölmesinde eğitir, görmediği konuşmacılardan oluşan test
bölmesinde değerlendirir ve models/ altına model + metadata yazar.

Kullanım:
    python scripts/train.py --task emergency
    python scripts/train.py --task all
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import sklearn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import FEATURES_DIR, MODELS_DIR, ROOT
from evaluation import format_summary, held_out_mask, load_features, make_model, summarize
from features import FEATURE_SPECS
from pipeline import MODEL_FILES, metadata_path

TASKS = ("emergency", "human")


def train(task):
    d = load_features(task)
    train_mask = d["split"] == "train"
    test_mask = held_out_mask(d)
    if not test_mask.any():
        sys.exit(f"[{task}] test bölmesi boş; manifest'i yeniden oluşturun.")

    labels = sorted(set(d["y"]))
    print(f"\n=== {task} ===")
    print(f"train: {train_mask.sum()} satır ({len(set(d['group'][train_mask]))} konuşmacı/grup), "
          f"test: {test_mask.sum()} kayıt ({len(set(d['group'][test_mask]))} konuşmacı/grup)")

    model = make_model(task)
    model.fit(d["X"][train_mask], d["y"][train_mask])
    summary = summarize(task, d["y"][test_mask], model.predict(d["X"][test_mask]), labels)
    print("\nGörülmemiş konuşmacılar üzerinde:\n" + format_summary(summary))

    with open(os.path.join(FEATURES_DIR, f"{task}.json"), encoding="utf-8") as f:
        features_meta = json.load(f)

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, MODEL_FILES[task])
    joblib.dump(model, path, compress=3)
    meta = {
        "task": task,
        "labels": [str(c) for c in model.classes_],
        "feature_spec": FEATURE_SPECS[task],
        "sklearn_version": sklearn.__version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": features_meta["manifest_sha256"],
        "augmentations": features_meta["augmentations"],
        "datasets": sorted(set(d["dataset"])),
        "n_train_rows": int(train_mask.sum()),
        "eval_protocol": "held-out speakers (dataset.assign_splits), original recordings only",
        "metrics": summary,
    }
    with open(metadata_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n-> {os.path.relpath(path, ROOT)} (+ .json)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS + ("all",), required=True)
    args = ap.parse_args()
    for task in TASKS if args.task == "all" else (args.task,):
        train(task)


if __name__ == "__main__":
    main()
