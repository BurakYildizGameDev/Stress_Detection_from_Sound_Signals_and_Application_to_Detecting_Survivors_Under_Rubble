import os
import shutil
import random

random.seed(42)

SOURCE = {
    "human": {
        "berlin": "data_sampled/human/berlin",
        "jl_corpus": "data_sampled/human/jl_corpus",
        "ravdess": "data_sampled/human/ravdess",
        "savee": "data_sampled/human/savee",
        "subesco": "data_sampled/human/subesco",
        "tess": "data_sampled/human/tess",
    },
    "non_human": {
        "esc50": "data_sampled/non_human/esc50",
        "urban": "data_sampled/non_human/urban",
    }
}

TARGET = "data_mid"

HUMAN_PER_CLASS = 1000
NON_HUMAN_PER_CLASS = 1500

def copy_samples(src, dst, limit):
    os.makedirs(dst, exist_ok=True)
    files = [f for f in os.listdir(src) if f.endswith(".wav")]
    random.shuffle(files)
    files = files[:limit]

    for f in files:
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))

    print(f"{dst}: {len(files)} dosya kopyalandı")

def main():
    for label in SOURCE:
        for name, path in SOURCE[label].items():
            limit = HUMAN_PER_CLASS if label == "human" else NON_HUMAN_PER_CLASS
            out_dir = os.path.join(TARGET, label, name)
            copy_samples(path, out_dir, limit)

    print("\n✅ ORTA DATASET HAZIR")

if __name__ == "__main__":
    main()
