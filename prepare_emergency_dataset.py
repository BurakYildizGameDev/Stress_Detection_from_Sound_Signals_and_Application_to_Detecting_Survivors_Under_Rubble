import os
import shutil
import random

SOURCE_ROOT = "data_final"        # ✅ GERÇEK KLASÖR
TARGET_ROOT = "data_emergency"

CLASSES = {
    "normal": ["neutral", "calm"],
    "stress": ["angry", "fearful"],
    "panic": ["fear", "panic"],
    "scream": ["scream", "shout"]
}

MAX_PER_CLASS = 500

def collect_files():
    os.makedirs(TARGET_ROOT, exist_ok=True)
    total = 0

    for target_class, keywords in CLASSES.items():
        out_dir = os.path.join(TARGET_ROOT, target_class)
        os.makedirs(out_dir, exist_ok=True)

        collected = []

        for root, _, files in os.walk(SOURCE_ROOT):
            for f in files:
                if f.lower().endswith(".wav"):
                    if any(k in f.lower() for k in keywords):
                        collected.append(os.path.join(root, f))

        random.shuffle(collected)
        selected = collected[:MAX_PER_CLASS]

        for f in selected:
            shutil.copy(f, out_dir)

        print(f"✅ {target_class}: {len(selected)} dosya taşındı")
        total += len(selected)

    print(f"\n✅ TOPLAM {total} dosya taşındı")

if __name__ == "__main__":
    collect_files()
