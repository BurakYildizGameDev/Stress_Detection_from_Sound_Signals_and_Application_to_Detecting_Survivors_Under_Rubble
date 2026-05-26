import os
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

def add_noise(data, noise_factor=0.005):
    """Gürültü ekle"""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data.astype(type(data[0]))

def time_shift(data, shift_max=0.2):
    """Zaman kaydırma"""
    shift = np.random.randint(int(len(data) * shift_max))
    direction = np.random.randint(0, 2)
    if direction == 1:
        augmented_data = np.roll(data, shift)
    else:
        augmented_data = np.roll(data, -shift)
    return augmented_data

def change_pitch(data, sr, pitch_factor=2):
    """Pitch (ton) değiştir"""
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_factor)

def change_speed(data, speed_factor=1.1):
    """Hız değiştir"""
    return librosa.effects.time_stretch(data, rate=speed_factor)

def augment_audio(file_path, output_dir, base_name, augmentations_per_file=2):
    """Tek ses dosyasını artır"""
    try:
        # Orijinal dosyayı yükle
        y, sr = librosa.load(file_path, sr=22050)
        
        # Orijinal dosyayı kaydet
        original_output = os.path.join(output_dir, f"{base_name}_original.wav")
        sf.write(original_output, y, sr)
        
        saved_count = 1
        
        # Augmentation 1: Gürültü ekle
        if augmentations_per_file >= 1:
            noisy = add_noise(y)
            noise_output = os.path.join(output_dir, f"{base_name}_noise.wav")
            sf.write(noise_output, noisy, sr)
            saved_count += 1
        
        # Augmentation 2: Pitch değiştir (yükselt)
        if augmentations_per_file >= 2:
            pitched_up = change_pitch(y, sr, pitch_factor=2)
            pitch_output = os.path.join(output_dir, f"{base_name}_pitch_up.wav")
            sf.write(pitch_output, pitched_up, sr)
            saved_count += 1
        
        # Augmentation 3: Pitch değiştir (alçalt)
        if augmentations_per_file >= 3:
            pitched_down = change_pitch(y, sr, pitch_factor=-2)
            pitch_down_output = os.path.join(output_dir, f"{base_name}_pitch_down.wav")
            sf.write(pitch_down_output, pitched_down, sr)
            saved_count += 1
        
        # Augmentation 4: Hız artır
        if augmentations_per_file >= 4:
            faster = change_speed(y, speed_factor=1.1)
            speed_up_output = os.path.join(output_dir, f"{base_name}_speed_up.wav")
            sf.write(speed_up_output, faster, sr)
            saved_count += 1
        
        # Augmentation 5: Hız azalt
        if augmentations_per_file >= 5:
            slower = change_speed(y, speed_factor=0.9)
            speed_down_output = os.path.join(output_dir, f"{base_name}_speed_down.wav")
            sf.write(speed_down_output, slower, sr)
            saved_count += 1
        
        return saved_count
    except Exception as e:
        print(f"Hata ({file_path}): {e}")
        return 0

def augment_dataset(input_dirs, output_dir, augmentations_per_file=2):
    """Tüm veri setini artır"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("VERİ ARTIRMA BAŞLIYOR")
    print("=" * 70)
    
    total_files = 0
    total_augmented = 0
    
    for dataset_name, dataset_path in input_dirs.items():
        if not os.path.exists(dataset_path):
            print(f"[{dataset_name}] Klasör bulunamadı, atlanıyor...")
            continue
        
        # Output klasörü oluştur
        dataset_output = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)
        
        print(f"\n[{dataset_name}] İşleniyor...")
        
        # Tüm wav dosyalarını topla
        wav_files = []
        if dataset_name in ['berlin', 'savee', 'jl_corpus', 'subesco']:
            # Düz klasör
            wav_files = [(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.wav')]
        elif dataset_name == 'ravdess':
            # Alt klasörler
            for actor in os.listdir(dataset_path):
                actor_path = os.path.join(dataset_path, actor)
                if os.path.isdir(actor_path):
                    for f in os.listdir(actor_path):
                        if f.endswith('.wav'):
                            wav_files.append((actor_path, f))
        elif dataset_name == 'tess':
            # Duygu klasörleri
            for folder in os.listdir(dataset_path):
                folder_path = os.path.join(dataset_path, folder)
                if os.path.isdir(folder_path):
                    for f in os.listdir(folder_path):
                        if f.endswith('.wav'):
                            wav_files.append((folder_path, f))
        
        print(f"  Toplam {len(wav_files)} dosya bulundu")
        
        # Her dosyayı augment et
        for i, (file_dir, filename) in enumerate(tqdm(wav_files, desc=f"  {dataset_name}")):
            file_path = os.path.join(file_dir, filename)
            base_name = filename.replace('.wav', '')
            
            count = augment_audio(file_path, dataset_output, base_name, augmentations_per_file)
            total_augmented += count
            
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(wav_files)} işlendi ({total_augmented} dosya)")
        
        total_files += len(wav_files)
        print(f"  [{dataset_name}] Tamamlandı!")
    
    print("\n" + "=" * 70)
    print(f"✅ VERİ ARTIRMA TAMAMLANDI!")
    print(f"  Orijinal dosya: {total_files}")
    print(f"  Artırılmış dosya: {total_augmented}")
    print(f"  Her dosya başına: {augmentations_per_file + 1} varyasyon")
    print(f"  Çıktı klasörü: {output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    # Veri setlerinin konumu
    input_directories = {
        'berlin': 'data/berlin',
        'ravdess': 'data/ravdess',
        'tess': 'data/tess',
        'savee': 'data/savee',
        'jl_corpus': 'data/jl_corpus',
        'subesco': 'data/subesco'
    }
    
    # Çıktı klasörü
    output_directory = 'data_augmented'
    
    # Her dosya için kaç varyasyon? (1-5 arası)
    # 2 = 3x artış (orijinal + 2 varyasyon)
    # 3 = 4x artış
    augmentations = 2
    
    print(f"Her dosya için {augmentations} augmentation yapılacak")
    print(f"Toplam artış: {augmentations + 1}x")
    
    augment_dataset(input_directories, output_directory, augmentations)