import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Berlin EMODB duygu kodları
BERLIN_EMOTION_TO_STRESS = {
    'W': 2,  # Öfke -> Yüksek Stres
    'L': 0,  # Sıkılma -> Düşük Stres
    'E': 1,  # İğrenme -> Orta Stres
    'A': 2,  # Korku -> Yüksek Stres
    'F': 0,  # Mutluluk -> Düşük Stres
    'T': 1,  # Üzüntü -> Orta Stres
    'N': 0   # Nötr -> Düşük Stres
}

# RAVDESS duygu kodları (3. segment)
RAVDESS_EMOTION_TO_STRESS = {
    '01': 0,  # Nötr -> Düşük Stres
    '02': 0,  # Sakin -> Düşük Stres
    '03': 0,  # Mutlu -> Düşük Stres
    '04': 1,  # Üzgün -> Orta Stres
    '05': 2,  # Öfkeli -> Yüksek Stres
    '06': 2,  # Korkulu -> Yüksek Stres
    '07': 1,  # İğrenme -> Orta Stres
    '08': 1   # Şaşkın -> Orta Stres
}

STRESS_LABELS = {0: 'Düşük Stres', 1: 'Orta Stres', 2: 'Yüksek Stres'}

def extract_features(file_path):
    """Ses dosyasından özellik çıkar"""
    try:
        y, sr = librosa.load(file_path, duration=3, sr=22050)
        
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        tempo = librosa.feature.tempo(y=y, sr=sr)[0]
        
        features = np.hstack([rms, zcr, spectral_centroid, spectral_rolloff, mfcc_mean, chroma, tempo])
        return features
    except Exception as e:
        print(f"Hata: {e}")
        return None

def load_berlin_data(data_dir):
    """Berlin EMODB veri setini yükle"""
    features_list = []
    labels_list = []
    
    berlin_dir = os.path.join(data_dir, 'berlin')
    
    if not os.path.exists(berlin_dir):
        print(f"UYARI: {berlin_dir} bulunamadı, atlanıyor...")
        return [], []
    
    wav_files = [f for f in os.listdir(berlin_dir) if f.endswith('.wav')]
    print(f"\n[Berlin EMODB] {len(wav_files)} dosya bulundu")
    
    for i, filename in enumerate(wav_files):
        try:
            emotion_code = filename[5]
            
            if emotion_code not in BERLIN_EMOTION_TO_STRESS:
                continue
            
            file_path = os.path.join(berlin_dir, filename)
            features = extract_features(file_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(BERLIN_EMOTION_TO_STRESS[emotion_code])
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(wav_files)} işlendi...")
        except Exception as e:
            continue
    
    print(f"[Berlin EMODB] {len(features_list)} örnek yüklendi")
    return features_list, labels_list

def load_ravdess_data(data_dir):
    """RAVDESS veri setini yükle"""
    features_list = []
    labels_list = []
    
    ravdess_dir = os.path.join(data_dir, 'ravdess')
    
    if not os.path.exists(ravdess_dir):
        print(f"UYARI: {ravdess_dir} bulunamadı, atlanıyor...")
        return [], []
    
    # Tüm Actor klasörlerini tara
    actor_folders = [f for f in os.listdir(ravdess_dir) if f.startswith('Actor_')]
    print(f"\n[RAVDESS] {len(actor_folders)} aktör klasörü bulundu")
    
    total_files = 0
    for actor_folder in actor_folders:
        actor_path = os.path.join(ravdess_dir, actor_folder)
        wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        
        for filename in wav_files:
            try:
                # Dosya formatı: 03-02-05-01-01-01-12.wav
                # 3. segment (index 2) = duygu kodu
                parts = filename.split('-')
                emotion_code = parts[2]
                
                if emotion_code not in RAVDESS_EMOTION_TO_STRESS:
                    continue
                
                file_path = os.path.join(actor_path, filename)
                features = extract_features(file_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(RAVDESS_EMOTION_TO_STRESS[emotion_code])
                
                total_files += 1
                if total_files % 100 == 0:
                    print(f"  {total_files} dosya işlendi...")
            except Exception as e:
                continue
    
    print(f"[RAVDESS] {len(features_list)} örnek yüklendi")
    return features_list, labels_list

def train_model():
    """Modeli eğit"""
    print("=" * 60)
    print("BİRLEŞİK VERİ SETİ İLE MODEL EĞİTİMİ")
    print("=" * 60)
    
    # Her iki veri setini yükle
    berlin_features, berlin_labels = load_berlin_data('data')
    ravdess_features, ravdess_labels = load_ravdess_data('data')
    
    if len(berlin_features) == 0 and len(ravdess_features) == 0:
        print("HATA: Hiç veri bulunamadı!")
        return
    
    # Birleştir
    all_features = berlin_features + ravdess_features
    all_labels = berlin_labels + ravdess_labels
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'='*60}")
    print(f"TOPLAM: {len(X)} örnek")
    print(f"  - Berlin EMODB: {len(berlin_features)} örnek")
    print(f"  - RAVDESS: {len(ravdess_features)} örnek")
    print(f"Özellik sayısı: {X.shape[1]}")
    
    # Sınıf dağılımı
    unique, counts = np.unique(y, return_counts=True)
    print("\nSınıf dağılımı:")
    for label, count in zip(unique, counts):
        print(f"  {STRESS_LABELS[label]}: {count} örnek (%{count/len(y)*100:.1f})")
    
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
        n_estimators=200,  # Daha fazla ağaç
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Değerlendirme
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print(f"MODEL DOĞRULUĞU: {accuracy * 100:.2f}%")
    print("=" * 60)
    print("\nDetaylı Rapor:")
    print(classification_report(
        y_test, y_pred,
        target_names=[STRESS_LABELS[i] for i in sorted(STRESS_LABELS.keys())]
    ))
    
    # Modeli kaydet
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stress_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✓ Model kaydedildi: models/stress_model.pkl")
    print("✓ Scaler kaydedildi: models/scaler.pkl")
    print("\nEski modelin üzerine yazıldı, app.py ile kullanabilirsin!")

if __name__ == "__main__":
    train_model()