"""
data/ altındaki veri setlerini tarayıp data/manifest.csv dosyasını üretir.

Her satır bir kayıttır: yol, veri seti, lisans, konuşmacı, kayıt kimliği, duygu,
acil durum sınıfı ve konuşmacı bazlı train/test bölmesi. Sonraki bütün adımlar
(öznitelik çıkarımı, eğitim, değerlendirme) bu dosyayı okur.

Kullanım:
    python scripts/build_manifest.py
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import (DATASETS, EMERGENCY_CLASSES, MANIFEST_PATH, ROOT,
                     assign_splits, dataset_dir, scan_dataset, write_manifest)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    records, missing = [], []
    for name in DATASETS:
        if not os.path.isdir(dataset_dir(name)):
            missing.append(name)
            continue
        recs, skipped = scan_dataset(name)
        print(f"[{name}] {len(recs)} kayıt")
        for reason, n in skipped.items():
            print(f"    atlandı: {n} ({reason})")
        records += recs

    if missing:
        print(f"\nBulunamayan veri setleri: {', '.join(missing)}")
        print("İndirmek için: python scripts/download_data.py " + " ".join(missing))
    if not records:
        sys.exit("Hiç kayıt bulunamadı; manifest yazılmadı.")

    assign_splits(records)
    write_manifest(records)

    print(f"\n{len(records)} kayıt -> {os.path.relpath(MANIFEST_PATH, ROOT)}\n")
    print("Acil durum sınıfları (train / test):")
    counts = Counter((r.emergency_class, r.split) for r in records if r.emergency_class)
    for c in EMERGENCY_CLASSES:
        print(f"  {c:8s} {counts[(c, 'train')]:5d} / {counts[(c, 'test')]:5d}")
    print("İnsan-sesi (train / test):")
    counts = Counter((r.role, r.split) for r in records)
    for role in ("human", "non_human"):
        print(f"  {role:9s} {counts[(role, 'train')]:5d} / {counts[(role, 'test')]:5d}")
    if not counts[("non_human", "train")]:
        print("\nUyarı: non_human kayıt yok, insan-sesi modeli eğitilemez "
              "(python scripts/download_data.py esc50).")


if __name__ == "__main__":
    main()
