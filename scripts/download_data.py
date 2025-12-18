"""
Ham veri setlerini indirir ve eğitim scriptlerinin beklediği klasörlere koyar.

Veri setleri lisansları gereği bu repoda dağıtılmaz. Bu script onları
orijinal kaynaklarından indirir.

Kullanım:
    python scripts/download_data.py              # indirilebilen hepsi
    python scripts/download_data.py ravdess tess # sadece seçilenler
    python scripts/download_data.py --list

Klasör yapısı ve lisanslar dataset.py'deki DATASETS tablosundan gelir:
    data/human/{berlin,ravdess,tess,subesco,savee,jl_corpus}/
    data/non-human/esc50/

Sonraki adım: python scripts/build_manifest.py
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import DATASETS as LAYOUT, ROOT, dataset_dir

TESS_DOI = "doi:10.5683/SP2/E8H2MF"
BOREALIS = "https://borealisdata.ca/api"

DATASETS = {
    "ravdess": {
        "url": "https://zenodo.org/records/1188976/files/Audio_Song_Actors_01-24.zip?download=1",
        "info": "RAVDESS (song alt kümesi), Livingstone & Russo 2018",
    },
    "berlin": {
        "url": "http://emodb.bilderbar.info/download/download.zip",
        "zip_subdir": "wav",
        "info": "Berlin EMO-DB, Burkhardt vd. 2005",
    },
    "subesco": {
        "url": "https://zenodo.org/records/4526477/files/SUBESCO.zip?download=1",
        "info": "SUST Bangla Emotional Speech Corpus (~1.7 GB)",
    },
    "tess": {
        "custom": "tess",
        "info": "Toronto Emotional Speech Set (Borealis, 2800 dosya)",
    },
    "esc50": {
        "url": "https://github.com/karoldvl/ESC-50/archive/master.zip",
        "zip_subdir": "audio",
        "info": "ESC-50 çevresel ses veri seti",
    },
    "savee": {
        "manual": "Kayıt gerektirir: http://kahlan.eps.surrey.ac.uk/savee/ "
                  "(veya Kaggle: ejlok1/surrey-audiovisual-expressed-emotion-savee). "
                  "wav dosyalarını data/human/savee/ altına koyun.",
        "info": "Surrey Audio-Visual Expressed Emotion",
    },
    "jl_corpus": {
        "manual": "Kaggle: tli725/jl-corpus. wav dosyalarını data/human/jl_corpus/ altına koyun.",
        "info": "JL-Corpus (Yeni Zelanda İngilizcesi)",
    },
}


def has_wavs(path):
    for _, _, files in os.walk(path):
        if any(f.lower().endswith(".wav") for f in files):
            return True
    return False


def download(url, path):
    last = [-1]

    def hook(blocks, bs, total):
        if total > 0:
            pct = min(100, blocks * bs * 100 // total)
            if pct != last[0]:
                last[0] = pct
                sys.stdout.write(f"\r  {pct:3d}%  ({total / 1e6:.0f} MB)")
                sys.stdout.flush()

    urllib.request.urlretrieve(url, path, hook)
    sys.stdout.write("\n")


def extract_wavs(zip_path, dest, subdir=None):
    """Zip içindeki wav dosyalarını klasör yapısını koruyarak dest'e çıkarır.
    subdir verilirse yalnızca o klasörün altındakiler alınır."""
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        names = [m for m in z.namelist()
                 if m.lower().endswith(".wav") and "__MACOSX" not in m]
        # Tüm dosyalar tek bir kök klasördeyse (ör. "SUBESCO/...") onu at
        roots = {m.split("/")[0] for m in names}
        strip_root = len(roots) == 1 and all("/" in m for m in names)
        for name in names:
            parts = name.split("/")
            if subdir:
                if subdir not in parts:
                    continue
                parts = parts[parts.index(subdir) + 1:]
            elif strip_root:
                parts = parts[1:]
            out = os.path.join(dest, *parts)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


def fetch_tess(dest):
    url = f"{BOREALIS}/datasets/:persistentId/?persistentId={TESS_DOI}"
    with urllib.request.urlopen(url) as r:
        files = json.load(r)["data"]["latestVersion"]["files"]
    os.makedirs(dest, exist_ok=True)
    for i, f in enumerate(files, 1):
        df = f["dataFile"]
        name = df["filename"]
        if not name.lower().endswith(".wav"):
            continue
        # OAF_back_angry.wav -> OAF_angry/OAF_back_angry.wav (orijinal düzen)
        speaker, _, rest = name.partition("_")
        emotion = os.path.splitext(rest)[0].split("_", 1)[-1]
        out = os.path.join(dest, f"{speaker}_{emotion}", name)
        if os.path.exists(out):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        urllib.request.urlretrieve(f"{BOREALIS}/access/datafile/{df['id']}", out)
        sys.stdout.write(f"\r  {i}/{len(files)}")
        sys.stdout.flush()
    sys.stdout.write("\n")


def fetch(name, spec, force=False):
    dest = dataset_dir(name)
    print(f"[{name}] {spec['info']} — lisans: {LAYOUT[name].license}")

    if has_wavs(dest) and not force:
        print(f"  zaten var: {os.path.relpath(dest, ROOT)} (yeniden indirmek için --force)")
        return True
    if "manual" in spec:
        print(f"  ELLE İNDİRİN: {spec['manual']}")
        return False
    if spec.get("custom") == "tess":
        fetch_tess(dest)
        return True

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, f"{name}.zip")
        download(spec["url"], zip_path)
        n = extract_wavs(zip_path, dest, spec.get("zip_subdir"))
    print(f"  {n} wav -> {os.path.relpath(dest, ROOT)}")
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="veri seti adları (boş = hepsi)")
    ap.add_argument("--list", action="store_true", help="veri setlerini listele")
    ap.add_argument("--force", action="store_true", help="var olsa da yeniden indir")
    args = ap.parse_args()

    if args.list:
        for k, v in DATASETS.items():
            kind = "elle" if "manual" in v else "otomatik"
            print(f"{k:10s} {kind:9s} {LAYOUT[k].license:28s} {v['info']}")
        return

    unknown = set(args.names) - set(DATASETS)
    if unknown:
        ap.error(f"bilinmeyen veri seti: {', '.join(sorted(unknown))}")

    missing = []
    for name in args.names or DATASETS:
        if not fetch(name, DATASETS[name], args.force):
            missing.append(name)

    if missing:
        print(f"\nElle indirilmesi gerekenler: {', '.join(missing)}")


if __name__ == "__main__":
    main()
