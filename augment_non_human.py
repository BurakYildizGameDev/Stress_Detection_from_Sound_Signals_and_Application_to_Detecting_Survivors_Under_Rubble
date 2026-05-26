import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# ===============================
# GÜVENLİ NON-HUMAN AUGMENTATION
# ===============================

def add_light_noise(data, noise_factor=0.002):
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def small_time_shift(data, shift_ratio=0.05):
    shift = int(len(data) * shift_ratio)
    shift = np.random.randint(-shift, shift)
    return np.roll(data, shift)

def volume_scale(data, min_gain=0.8, max_gain=1.2):
    gain = np.random.uniform(min_gain, max_gain)
    return data * gain

def augment_non_human(file_path, output_dir, base_name):
    try:
        y, sr = librosa.load(file_path, sr=22050)

        count = 0

        # 1️⃣ Orijinal
        sf.write(os.path.join(output_dir, f"{base_name}_orig.wav"), y, sr)
        count += 1

        # 2️⃣ Hafif gürültü
        noisy = add_light_noise(y)
        sf.write(os.path.join(output_dir, f"{base_name}_noise.wav"), noisy, sr)
        count += 1

        # 3️⃣ Küçük time shift
        shifted = small_time_shift(y)
        sf.write(os.path.join(output_dir, f"{base_name}_shift.wav"), shifted, sr)
        count += 1

        # 4️⃣ Hafif volume scaling
        scaled = volume_scale(y)
        sf.write(os.path.join(output_dir, f"{base_name}_volume.wav"), scaled, sr)
        count += 1

        return count

    except Exception as e:
        print(f"Hata: {file_path} -> {e}")
        return 0

# ===============================
# DATASET TARAYICI
# ===============================

def process_dataset(input_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    total_files = 0
    total_outputs = 0

    print("\nNON-HUMAN DATA AUGMENTATION BAŞLADI\n")

    for name, path in input_dirs.items():
        if not os.path.exists(path):
            print(f"[{name}] bulunamadı, atlandı")
            continue

        out_path = os.path.join(output_dir, name)
        os.makedirs(out_path, exist_ok=True)

        print(f"[{name}] işleniyor...")

        audio_files = []
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".wav") or f.endswith(".mp3"):
                    audio_files.append(os.path.join(root, f))

        print(f"  {len(audio_files)} dosya bulundu")

        for file_path in tqdm(audio_files):
            base = os.path.splitext(os.path.basename(file_path))[0]
            total_outputs += augment_non_human(file_path, out_path, base)

        total_files += len(audio_files)

    print("\n===================================")
    print("NON-HUMAN AUGMENTATION TAMAMLANDI")
    print(f"Orijinal dosya sayısı : {total_files}")
    print(f"Oluşan dosya sayısı   : {total_outputs}")
    print("===================================")

# ===============================
# ÇALIŞTIRMA
# ===============================

if __name__ == "__main__":
    input_directories = {
        "esc50": "data/non-human/esc50",
        "urban": "data/non-human/urban"
    }

    output_directory = "data_augmented_non_human"

    process_dataset(input_directories, output_directory)
