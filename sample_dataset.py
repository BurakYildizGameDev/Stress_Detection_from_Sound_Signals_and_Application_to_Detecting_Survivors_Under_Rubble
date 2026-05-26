import os
import random
import shutil

# kaç dosya alınacak
HUMAN_PER_DATASET = 300
NON_HUMAN_PER_FOLDER = 300

SRC = "data_final"
DST = "data_sampled"

HUMAN_SRC = os.path.join(SRC, "human")
NON_HUMAN_SRC = os.path.join(SRC, "non_human")

random.seed(42)

def copy_samples(src_dir, dst_dir, max_files):
    os.makedirs(dst_dir, exist_ok=True)

    wavs = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".wav"):
                wavs.append(os.path.join(root, f))

    random.shuffle(wavs)
    wavs = wavs[:max_files]

    for f in wavs:
        shutil.copy(f, os.path.join(dst_dir, os.path.basename(f)))

    print(f"{dst_dir}: {len(wavs)} dosya kopyalandı")

# HUMAN
for ds in os.listdir(HUMAN_SRC):
    src = os.path.join(HUMAN_SRC, ds)
    dst = os.path.join(DST, "human", ds)
    copy_samples(src, dst, HUMAN_PER_DATASET)

# NON-HUMAN
for ds in os.listdir(NON_HUMAN_SRC):
    src = os.path.join(NON_HUMAN_SRC, ds)
    dst = os.path.join(DST, "non_human", ds)
    copy_samples(src, dst, NON_HUMAN_PER_FOLDER)

print("\n✅ SAMPLE DATASET HAZIR")
