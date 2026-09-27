"""
extract_features_v2.py - Fısıltı ve inleme destekli v2 öznitelik çıkarımı.

Satır üretimi:
- Her konuşma kaydından (eval_only olmayan veri setleri) deterministik olarak
  ya bir fısıltı ya da bir inleme sentezlenir; dönüştürücü parametreleri
  rastgeledir. Sentez hem train hem test konuşmacılarına uygulanır, böylece
  whisper/moan sınıfları görülmemiş konuşmacılar üzerinde ölçülebilir.
- train: orijinal + genel artırmalar + rastgele enkaz (random_rubble).
  Enkaz artırması sentezlenen fısıltı/inlemeye de uygulanır; aksi hâlde model
  "enkaz izi varsa fısıltı değildir" kısayolunu öğrenir.
- test: her kayıt (ve sentezi) sabit koşullarda: clean ve
  rubble_physical_{mild,medium,severe} (augment.CONDITIONS). Test satırları
  hiçbir zaman eğitimde kullanılmaz.
- eval_only veri setleri (VIVAE, field) yalnızca test satırı üretir ve
  sentez kaynağı olarak kullanılmaz: bunlar gerçek dış doğrulama verisidir.

Sütunlar: X, y, split, group, dataset, path, transform (eğitim artırması),
synth (none | dsp_whisper | dsp_moan), condition (train | test koşulu).

Kullanım:
    python scripts/extract_features_v2.py --task all
"""
import argparse
import json
import os
import random
import sys
import zlib
from datetime import datetime, timezone

import numpy as np
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augment import AUGMENTATIONS, CONDITIONS
from augment_rubble import random_rubble
from dataset import (EVAL_ONLY_DATASETS, FEATURES_DIR, MANIFEST_PATH, ROOT,
                     SPLIT_SEED, file_sha256, read_manifest)
from features_v2 import FEATURE_FUNCS_V2, FEATURE_SPECS_V2
from whisper_converter import (convert_to_moan_dsp, convert_to_whisper_dsp,
                               random_moan_params, random_whisper_params)

TASKS = ("emergency", "human")
TEST_CONDITIONS = ("clean", "rubble_physical_mild", "rubble_physical_medium",
                   "rubble_physical_severe")
SPEECH_CLASSES = ("normal", "stress", "panic")


def features_path_v2(task):
    return os.path.join(FEATURES_DIR, f"{task}_v2.npz")


def select_rows_v2(task, records, max_per_dataset=300, seed=SPLIT_SEED):
    """Göreve giren kayıtlar ve etiketleri."""
    if task == "emergency":
        return [(r, r.emergency_class_v2) for r in records if r.emergency_class_v2]

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


def synth_source(task, rec):
    """Bu kayıttan fısıltı/inleme sentezlenir mi?"""
    if rec.dataset in EVAL_ONLY_DATASETS:
        return False
    if task == "emergency":
        return rec.emergency_class_v2 in SPEECH_CLASSES
    return rec.role == "human"


def synthesize(y, sr, kind, rng):
    if kind == "dsp_whisper":
        return convert_to_whisper_dsp(y, sr, **random_whisper_params(rng))
    return convert_to_moan_dsp(y, sr, **random_moan_params(rng))


def process_row_v2(task, rec, label, augment):
    """Bir kayıt -> [(öznitelik, transform, synth, condition, etiket), ...]."""
    spec = FEATURE_SPECS_V2[task]
    func = FEATURE_FUNCS_V2[task]
    sr = spec["sr"]
    try:
        y, _ = librosa.load(os.path.join(ROOT, rec.path), sr=sr, duration=spec["clip_sec"])
    except Exception as e:
        return None, f"{rec.path}: {e}"
    if len(y) == 0:
        return None, f"Boş ses: {rec.path}"

    def rng_for(tag):
        return np.random.default_rng(zlib.crc32(f"{rec.path}|v{spec['version']}|{tag}".encode()))

    # (synth adı, sinyal, etiket)
    sources = [("none", y, label)]
    if synth_source(task, rec):
        kind = "dsp_whisper" if zlib.crc32(rec.path.encode()) % 2 == 0 else "dsp_moan"
        synth_label = kind[4:] if task == "emergency" else "human"
        sources.append((kind, synthesize(y, sr, kind, rng_for(kind)), synth_label))

    out = []
    if rec.split == "train":
        for synth, sig, lab in sources:
            out.append((func(sig, sr), "original", synth, "train", lab))
            if not augment:
                continue
            out.append((func(random_rubble(sig, sr, rng_for(f"{synth}|rubble")), sr),
                        "rubble_random", synth, "train", lab))
            if synth == "none":
                role = "human" if task == "emergency" else rec.role
                for name, fn in AUGMENTATIONS[role].items():
                    out.append((func(fn(sig, sr, rng_for(name)), sr), name, synth, "train", lab))
    else:
        for synth, sig, lab in sources:
            for cond in TEST_CONDITIONS:
                sig_c = CONDITIONS[cond](sig, sr, rng_for(f"{synth}|{cond}"))
                out.append((func(sig_c, sr), "original", synth, cond, lab))
    return out, None


def extract_task_v2(task, augment=True, max_per_dataset=300, n_jobs=-1):
    manifest = read_manifest(MANIFEST_PATH)
    rows = select_rows_v2(task, manifest, max_per_dataset)
    print(f"\n[*] {task}_v2 | kayıt: {len(rows)} | artırma: {augment}")

    jobs = (delayed(process_row_v2)(task, r, lab, augment) for r, lab in rows)
    cols = {k: [] for k in ("X", "y", "split", "group", "dataset", "path",
                            "transform", "synth", "condition")}
    errors = 0
    for (results, err), (rec, _) in tqdm(
            zip(Parallel(n_jobs=n_jobs, return_as="generator")(jobs), rows), total=len(rows)):
        if err:
            errors += 1
            continue
        for feat, transform, synth, cond, label in results:
            for k, v in (("X", feat), ("y", label), ("split", rec.split), ("group", rec.group),
                         ("dataset", rec.dataset), ("path", rec.path), ("transform", transform),
                         ("synth", synth), ("condition", cond)):
                cols[k].append(v)
    if errors:
        print(f"[!] {errors} kayıt okunamadı.")

    arrays = {k: (np.stack(v).astype(np.float32) if k == "X" else np.array(v))
              for k, v in cols.items()}
    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_npz = features_path_v2(task)
    np.savez_compressed(out_npz, **arrays)

    meta = {
        "task": f"{task}_v2",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "feature_spec": FEATURE_SPECS_V2[task],
        "n_samples": int(len(arrays["X"])),
        "classes": sorted(set(arrays["y"].tolist())),
        "augmented": bool(augment),
        "max_per_dataset": max_per_dataset if task == "human" else None,
        "test_conditions": list(TEST_CONDITIONS),
        "eval_only_datasets": sorted(EVAL_ONLY_DATASETS & set(arrays["dataset"].tolist())),
    }
    with open(os.path.splitext(out_npz)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[+] {os.path.relpath(out_npz, ROOT)} {arrays['X'].shape}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=TASKS + ("all",), default="all")
    p.add_argument("--no-augment", action="store_true", help="Artırmayı kapat")
    p.add_argument("--max-per-dataset", type=int, default=300,
                   help="insan-sesi görevinde veri seti başına en fazla insan kaydı")
    p.add_argument("--jobs", type=int, default=-1)
    args = p.parse_args()
    for t in TASKS if args.task == "all" else (args.task,):
        extract_task_v2(t, not args.no_augment, args.max_per_dataset, args.jobs)


if __name__ == "__main__":
    main()
