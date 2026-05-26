import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Duygu etiketleri
EMOTION_LABELS = {
    0: 'Düşük Stres',
    1: 'Orta Stres', 
    2: 'Yüksek Stres'
}

def extract_advanced_features(file_path):
    """Gelişmiş özellik çıkarma - 40+ özellik"""
    try:
        y, sr = librosa.load(file_path, duration=3, sr=22050)
        
        # 1. RMS Energy
        rms = np.mean(librosa.feature.rms(y=y))
        rms_std = np.std(librosa.feature.rms(y=y))
        
        # 2. Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        zcr_std = np.std(librosa.feature.zero_crossing_rate(y))
        
        # 3. Spectral Centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_centroid_std = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
        
        # 4. Spectral Rolloff
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_rolloff_std = np.std(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        # 5. Spectral Bandwidth
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_bandwidth_std = np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        # 6. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast)
        spectral_contrast_std = np.std(spectral_contrast)
        
        # 7. Spectral Flatness
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # 8. MFCC (20 coefficient)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 9. Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma)
        chroma_std = np.std(chroma)
        
        # 10. Tempo
        tempo = librosa.feature.tempo(y=y, sr=sr)[0]
        
        # Tüm özellikleri birleştir
        features = np.hstack([
            # Energy features (3)
            rms, rms_std, np.max(librosa.feature.rms(y=y)),
            
            # ZCR features (2)
            zcr, zcr_std,
            
            # Spectral features (9)
            spectral_centroid, spectral_centroid_std,
            spectral_rolloff, spectral_rolloff_std,
            spectral_bandwidth, spectral_bandwidth_std,
            spectral_contrast_mean, spectral_contrast_std,
            spectral_flatness,
            
            # MFCC features (20 mean + 20 std = 40)
            mfcc_mean, mfcc_std,
            
            # Chroma features (2)
            chroma_mean, chroma_std,
            
            # Tempo (1)
            tempo
        ])
        
        return features
    except Exception as e:
        return None

def get_emotion_from_filename(filename, dataset_name):
    """Dosya adından duygu etiketini çıkar"""
    try:
        if dataset_name == 'berlin':
            # 03a01Fa.wav -> F (6. karakter)
            emotion = filename[5]
            mapping = {'W': 2, 'L': 0, 'E': 1, 'A': 2, 'F': 0, 'T': 1, 'N': 0}
            
        elif dataset_name == 'ravdess':
            # 03-02-05-01-01-01-12_*.wav
            parts = filename.split('-')
            emotion = parts[2] if len(parts) > 2 else None
            mapping = {'01': 0, '02': 0, '03': 0, '04': 1, '05': 2, '06': 2, '07': 1, '08': 1}
            
        elif dataset_name == 'tess':
            # OAF_angry_*.wav
            if 'angry' in filename or 'fear' in filename:
                return 2
            elif 'sad' in filename or 'disgust' in filename:
                return 1
            else:
                return 0
                
        elif dataset_name == 'savee':
            # DC_a01_*.wav
            parts = filename.split('_')
            if len(parts) < 2:
                return None
            emotion = parts[1][0:2] if parts[1].startswith('sa') else parts[1][0]
            mapping = {'a': 2, 'd': 1, 'f': 2, 'h': 0, 'n': 0, 'sa': 1, 'su': 0}
        else:
            return None
            
        return mapping.get(emotion, None)
    except:
        return None

def load_augmented_data(data_dir):
    """Artırılmış veriyi yükle"""
    features_list = []
    labels_list = []
    
    datasets = ['berlin', 'ravdess', 'tess', 'savee']
    
    print("=" * 70)
    print("GELİŞMİŞ ÖZELLİK ÇIKARMA - 40+ ÖZELLIK")
    print("=" * 70)
    
    for dataset in datasets:
        dataset_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(dataset_dir):
            print(f"\n[{dataset.upper()}] Klasör bulunamadı, atlanıyor...")
            continue
        
        wav_files = [f for f in os.listdir(dataset_dir) if f.endswith('.wav')]
        print(f"\n[{dataset.upper()}] {len(wav_files)} dosya bulundu")
        
        for i, filename in enumerate(wav_files):
            try:
                # Duygu etiketi
                label = get_emotion_from_filename(filename, dataset)
                if label is None:
                    continue
                
                # Özellik çıkar
                file_path = os.path.join(dataset_dir, filename)
                features = extract_advanced_features(file_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{len(wav_files)} işlendi...")
            except:
                continue
        
        print(f"[{dataset.upper()}] {sum(1 for l in labels_list[-len(wav_files):])} örnek yüklendi")
    
    return np.array(features_list), np.array(labels_list)

def train_model():
    """Model eğit"""
    print("\n" + "=" * 70)
    print("MODEL EĞİTİMİ BAŞLIYOR")
    print("=" * 70)
    
    # Veriyi yükle
    X, y = load_augmented_data('data_augmented')
    
    if len(X) == 0:
        print("HATA: Veri bulunamadı!")
        return
    
    print(f"\nToplam örnek: {len(X)}")
    print(f"Özellik sayısı: {X.shape[1]}")
    
    # Sınıf dağılımı
    unique, counts = np.unique(y, return_counts=True)
    print("\nSınıf Dağılımı:")
    for label, count in zip(unique, counts):
        print(f"  {EMOTION_LABELS[label]}: {count} (%{count/len(y)*100:.1f})")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizasyon
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nModel eğitiliyor... (bu 10-15 dakika sürebilir)")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    model.fit(X_train, y_train)
    
    # Değerlendirme
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print(f"🎉 MODEL DOĞRULUĞU: {accuracy * 100:.2f}% 🎉")
    print("=" * 70)
    print("\nDetaylı Rapor:")
    print(classification_report(
        y_test, y_pred,
        target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())]
    ))
    
    # Modeli kaydet
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stress_model_advanced.pkl')
    joblib.dump(scaler, 'models/scaler_advanced.pkl')
    
    print(f"\n✓ Model kaydedildi: models/stress_model_advanced.pkl")
    print(f"✓ Scaler kaydedildi: models/scaler_advanced.pkl")
    print(f"✓ {len(X)} örnek ile eğitildi")
    print(f"✓ {X.shape[1]} özellik kullanıldı")

if __name__ == "__main__":
    train_model()