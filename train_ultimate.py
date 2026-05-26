import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Duygu etiketleri
EMOTION_LABELS = {0: 'Düşük Stres', 1: 'Orta Stres', 2: 'Yüksek Stres'}

# Veri seti etiketleme
DATASET_EMOTIONS = {
    'berlin': {'W': 2, 'L': 0, 'E': 1, 'A': 2, 'F': 0, 'T': 1, 'N': 0},
    'ravdess': {'01': 0, '02': 0, '03': 0, '04': 1, '05': 2, '06': 2, '07': 1, '08': 1},
    'tess': {'angry': 2, 'fear': 2, 'disgust': 1, 'sad': 1, 'happy': 0, 'neutral': 0, 'ps': 0, 'pleasant_surprise': 0},
    'savee': {'a': 2, 'd': 1, 'f': 2, 'h': 0, 'n': 0, 'sa': 1, 'su': 0},
    'jl_corpus': {'angry': 2, 'sad': 1, 'happy': 0, 'neutral': 0, 'excited': 0},
    'subesco': {'angry': 2, 'fear': 2, 'disgust': 1, 'sad': 1, 'happy': 0, 'neutral': 0, 'surprise': 0}
}

def extract_advanced_features(file_path):
    """Gelişmiş özellik çıkarma - 57 özellik"""
    try:
        y, sr = librosa.load(file_path, duration=3, sr=22050)
        
        # Energy features
        rms = np.mean(librosa.feature.rms(y=y))
        rms_std = np.std(librosa.feature.rms(y=y))
        rms_max = np.max(librosa.feature.rms(y=y))
        
        # ZCR
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        zcr_std = np.std(librosa.feature.zero_crossing_rate(y))
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_centroid_std = np.std(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_rolloff_std = np.std(librosa.feature.spectral_rolloff(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_bandwidth_std = np.std(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast)
        spectral_contrast_std = np.std(spectral_contrast)
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
        
        # MFCC (20)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma)
        chroma_std = np.std(chroma)
        
        # Tempo
        tempo = librosa.feature.tempo(y=y, sr=sr)[0]
        
        features = np.hstack([
            rms, rms_std, rms_max,
            zcr, zcr_std,
            spectral_centroid, spectral_centroid_std,
            spectral_rolloff, spectral_rolloff_std,
            spectral_bandwidth, spectral_bandwidth_std,
            spectral_contrast_mean, spectral_contrast_std,
            spectral_flatness,
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            tempo
        ])
        
        return features
    except:
        return None

def get_emotion_from_filename(filename, dataset_name):
    """Dosya adından duygu etiketini çıkar"""
    try:
        filename_lower = filename.lower()
        
        if dataset_name == 'berlin':
            emotion = filename[5]
            return DATASET_EMOTIONS['berlin'].get(emotion, None)
            
        elif dataset_name == 'ravdess':
            parts = filename.split('-')
            emotion = parts[2] if len(parts) > 2 else None
            return DATASET_EMOTIONS['ravdess'].get(emotion, None)
            
        elif dataset_name == 'tess':
            for emotion in DATASET_EMOTIONS['tess'].keys():
                if emotion in filename_lower:
                    return DATASET_EMOTIONS['tess'][emotion]
            return None
                
        elif dataset_name == 'savee':
            parts = filename.split('_')
            if len(parts) < 2:
                return None
            emotion = parts[1][0:2] if parts[1].startswith('sa') else parts[1][0]
            return DATASET_EMOTIONS['savee'].get(emotion, None)
            
        elif dataset_name == 'jl_corpus':
            for emotion in DATASET_EMOTIONS['jl_corpus'].keys():
                if emotion in filename_lower:
                    return DATASET_EMOTIONS['jl_corpus'][emotion]
            return None
            
        elif dataset_name == 'subesco':
            for emotion in DATASET_EMOTIONS['subesco'].keys():
                if emotion in filename_lower:
                    return DATASET_EMOTIONS['subesco'][emotion]
            return None
        
        return None
    except:
        return None

def load_dataset(data_dir):
    """Tüm veri setlerini yükle"""
    features_list = []
    labels_list = []
    
    datasets = ['berlin', 'ravdess', 'tess', 'savee', 'jl_corpus']
    
    print("=" * 70)
    print("TÜM VERİ SETLERİ YÜKLENİYOR (AUGMENTED)")
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
                label = get_emotion_from_filename(filename, dataset)
                if label is None:
                    continue
                
                file_path = os.path.join(dataset_dir, filename)
                features = extract_advanced_features(file_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                
                if (i + 1) % 2000 == 0:
                    print(f"  {i + 1}/{len(wav_files)} işlendi...")
            except:
                continue
        
        print(f"[{dataset.upper()}] Tamamlandı - {sum(1 for _ in labels_list[-len(wav_files):])} örnek yüklendi")
    
    return np.array(features_list), np.array(labels_list)

def train_model():
    """Model eğit"""
    print("\n" + "=" * 70)
    print("ULTIMATE MODEL EĞİTİMİ - TÜM VERİ SETLERİ")
    print("=" * 70)
    
    # Veriyi yükle
    X, y = load_dataset('data_augmented')
    
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
    
    print("\nModel eğitiliyor...")
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
    joblib.dump(model, 'models/stress_model_ultimate.pkl')
    joblib.dump(scaler, 'models/scaler_ultimate.pkl')
    
    print(f"\n✓ Model kaydedildi: models/stress_model_ultimate.pkl")
    print(f"✓ Scaler kaydedildi: models/scaler_ultimate.pkl")
    print(f"✓ {len(X)} örnek ile eğitildi")
    print(f"✓ {X.shape[1]} özellik kullanıldı")

if __name__ == "__main__":
    train_model()