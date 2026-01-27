"""
Eğitime hazırlığı tek komutla, sırayla çalıştırır:

  1. scripts/download_data.py      eksik veri setlerini indir (--skip-download ile atlanır)
  2. scripts/build_manifest.py     data/manifest.csv
  3. scripts/extract_features.py   features/{emergency,human}.npz
  4. scripts/train.py              yalnızca --train verilirse
  5. scripts/evaluate.py           yalnızca --train verilirse (tüm protokoller)

Her adım bir öncekinin çıktısını okur; bir adım hata verirse akış durur.

Kullanım:
    python scripts/prepare_training.py
    python scripts/prepare_training.py --skip-download --train
"""
import argparse
import os
import subprocess
import sys

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def step(name, *args):
    print(f"\n>>> {name} {' '.join(args)}", flush=True)
    code = subprocess.call([sys.executable, os.path.join(SCRIPTS, name), *args])
    if code:
        sys.exit(f"{name} başarısız oldu (çıkış kodu {code}); akış durduruldu.")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--train", action="store_true", help="modelleri eğit ve değerlendir")
    args = ap.parse_args()

    if not args.skip_download:
        step("download_data.py")
    step("build_manifest.py")
    step("extract_features.py", "--task", "all", *(["--no-augment"] if args.no_augment else []))
    if args.train:
        step("train.py", "--task", "all")
        step("evaluate.py", "--task", "all", "--protocol", "all")
    else:
        print("\nEğitime hazır. Eğitmek için: python scripts/train.py --task all")


if __name__ == "__main__":
    main()
