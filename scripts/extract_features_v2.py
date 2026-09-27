"""
extract_features_v2.py - Fısıltı ve İnleme Destekli V2 Öznitelik Çıkarım Scripti.

Bu script:
1. features_v2 modülünü (34-boyutlu insan, 20-boyutlu acil durum) kullanır.
2. Eğitim (train) kümesindeki konuşma örneklerinden LPC tabanlı fısıltı ve
   derin inleme sentezleyerek 'whisper' ve 'moan' sınıflarını otomatik artırır.
3. Çıktıları features/<task>_v2.npz ve .json dosyalarına yazar.

Kullanım:
    python scripts/extract_features_v2.py --task emergency
    python scripts/extract_features_v2.py --task human
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

from augment import AUGMENTATIONS
from dataset import (FEATURES_DIR, MANIFEST_PATH, ROOT, SPLIT_SEED,
                     file_sha256, read_manifest, EMERGENCY_CLASSES_V2)
from features_v2 import FEATURE_FUNCS_V2, FEATURE_SPECS_V2
from whisper_converter import convert_to_whisper_dsp, convert_to_moan_dsp
from augment_rubble import augment_with_rubble

TASKS = ("emergency", "human")


def features_path_v2(task):
    return os.path.join(FEATURES_DIR, f"{task}_v2.npz")


def select_rows_v2(task, records, max_per_dataset=None, seed=SPLIT_SEED):
    """Göreve giren kayıtlar ve etiketleri."""
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


def process_row_v2(task, rec, augment):
    """Bir kayıt -> [(öznitelik, dönüşüm adı, etiket), ...]."""
    spec = FEATURE_SPECS_V2[task]
    func = FEATURE_FUNCS_V2[task]
    abs_path = os.path.join(ROOT, rec.path.replace("/", os.sep))

    if not os.path.exists(abs_path):
        return None, f"Dosya yok: {abs_path}"

    try:
        y, orig_sr = librosa.load(abs_path, sr=spec["sr"])
    except Exception as e:
        return None, f"{abs_path}: {e}"

    if len(y) == 0:
        return None, f"Boş ses: {abs_path}"

    if spec["clip_sec"]:
        y = y[:int(spec["clip_sec"] * spec["sr"])]

    # Deterministik RNG
    seed = zlib.crc32(f"{rec.path}-{spec['version']}".encode("utf-8"))
    rng = np.random.default_rng(seed)

    base_feat = func(y, spec["sr"])
    default_label = rec.emergency_class if task == "emergency" else rec.role
    results = [(base_feat, "original", default_label)]

    # Yalnızca train kümesindeki verilere artırma uygulanır
    if augment and rec.split == "train":
        if task == "human":
            for name, aug_fn in AUGMENTATIONS.get(rec.role, {}).items():
                y_aug = aug_fn(y, spec["sr"], rng)
                results.append((func(y_aug, spec["sr"]), name, default_label))
            # Fısıltı ve inleme insan sesidir (human)
            if rec.role == "human":
                y_wh = convert_to_whisper_dsp(y, spec["sr"])
                results.append((func(y_wh, spec["sr"]), "dsp_whisper", "human"))
                y_mn = convert_to_moan_dsp(y, spec["sr"])
                results.append((func(y_mn, spec["sr"]), "dsp_moan", "human"))

            # 2. Aşama: Gerçek Enkaz Akustiği ve RIR Konvolüsyonu Artırması
            y_rubble = augment_with_rubble(y, spec["sr"], condition="rubble_medium", rng=rng)
            results.append((func(y_rubble, spec["sr"]), "rubble_medium", default_label))
        elif task == "emergency":
            # Acil durum için genel artırmalar
            for name, aug_fn in AUGMENTATIONS.get("human", {}).items():
                y_aug = aug_fn(y, spec["sr"], rng)
                results.append((func(y_aug, spec["sr"]), name, default_label))
            # Konuşma stres/panik kayıtlarından whisper ve moan türet
            if rec.emergency_class in ("stress", "panic"):
                y_wh = convert_to_whisper_dsp(y, spec["sr"])
                results.append((func(y_wh, spec["sr"]), "dsp_whisper", "whisper"))
                y_mn = convert_to_moan_dsp(y, spec["sr"])
                results.append((func(y_mn, spec["sr"]), "dsp_moan", "moan"))
            # Enkaz moloz distorsiyonu artırması
            y_rubble = augment_with_rubble(y, spec["sr"], condition="rubble_medium", rng=rng)
            results.append((func(y_rubble, spec["sr"]), "rubble_medium", default_label))

    return results, None


def extract_task_v2(task, augment=True, max_per_dataset=None, n_jobs=-1):
    manifest = read_manifest(MANIFEST_PATH)
    rows = select_rows_v2(task, manifest, max_per_dataset)

    print(f"\n[*] Görev: {task}_v2 | Toplam Kayıt: {len(rows)} | Artırma: {augment}")
    pool = Parallel(n_jobs=n_jobs, return_as="generator")
    jobs = (delayed(process_row_v2)(task, r, augment) for r, _ in rows)

    X_list, y_list, split_list, group_list, trans_list, ds_list = [], [], [], [], [], []
    errors = 0

    for (results, err), (rec, _) in tqdm(zip(pool(jobs), rows), total=len(rows)):
        if err:
            errors += 1
            continue
        for feat, transform, label in results:
            X_list.append(feat)
            y_list.append(label)
            split_list.append(rec.split)
            group_list.append(rec.group)
            trans_list.append(transform)
            ds_list.append(rec.dataset)

    if errors:
        print(f"[!] {errors} kayıt okunamadı.")

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list)
    split = np.array(split_list)
    group = np.array(group_list)
    transform = np.array(trans_list)
    dataset_arr = np.array(ds_list)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_npz = features_path_v2(task)
    np.savez_compressed(
        out_npz,
        X=X, y=y, split=split, group=group, transform=transform, dataset=dataset_arr
    )

    meta = {
        "task": f"{task}_v2",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "feature_spec": FEATURE_SPECS_V2[task],
        "n_samples": int(len(X)),
        "classes": sorted(list(set(y))),
        "augmented": bool(augment),
    }
    with open(os.path.splitext(out_npz)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[+] Başarıyla kaydedildi: {out_npz} (Şekil: {X.shape})")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="V2 Öznitelik Çıkarımı")
    p.add_argument("--task", choices=TASKS + ("all",), default="all")
    p.add_argument("--no-augment", action="store_true", help="Artırmayı kapat")
    p.add_argument("--max-per-dataset", type=int, default=None)
    args = p.parse_args()

    tasks = TASKS if args.task == "all" else (args.task,)
    for t in tasks:
        extract_task_v2(t, augment=not args.no_augment, max_per_dataset=args.max_per_dataset)


if __name__ == "__main__":
    main()
