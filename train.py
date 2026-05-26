import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Duygu kodlarını stres seviyesine çevir
EMOTION_TO_STRESS = {
    'W': 2,  # Öfke (Wut) -> Yüksek Stres
    'L': 0,  # Sıkılma (Langeweile) -> Düşük Stres
    'E': 1,  # İğrenme (Ekel) -> Orta Stres
    'A': 2,  # Korku (Angst) -> Yüksek Stres
    'F': 0,  # Mutluluk (Freude) -> Düşük Stres
    'T': 1,  # Üzüntü (Trauer) -> Orta Stres
    'N': 0   # Nötr -> Düşük Stres
}

STRESS_LABELS = {0: 'Düşük Stres', 1: 'Orta Stres', 2: 'Yüksek Stres'}

def extract_features(file_path):
    """Ses dosyasından özellik çıkar"""
    try:
        y, sr = librosa.load(file_path, duration=3, sr=22050)
        
        # Özellikler
        rms = np.mean(librosa.feature.rms(y=y))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        tempo = librosa.beat.tempo(y=y, sr=sr)[0]
        
        features = np.hstack([rms, zcr, spectral_centroid, spectral_rolloff, mfcc_mean, chroma, tempo])
        return features
    except Exception as e:
        print(f"Hata: {e}")
        return None

def load_data(data_dir):
    """Veri setini yükle"""
    features_list = []
    labels_list = []
    
    wav_dir = os.path.join(data_dir, 'wav')
    
    if not os.path.exists(wav_dir):
        print(f"HATA: {wav_dir} bulunamadı!")
        return None, None
    
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    print(f"{len(wav_files)} ses dosyası bulundu. İşleniyor...")
    
    for i, filename in enumerate(wav_files):
        try:
            emotion_code = filename[5]  # Dosya adından duygu kodu
            
            if emotion_code not in EMOTION_TO_STRESS:
                continue
            
            file_path = os.path.join(wav_dir, filename)
            features = extract_features(file_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(EMOTION_TO_STRESS[emotion_code])
            
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(wav_files)} dosya işlendi...")
        except Exception as e:
            print(f"Dosya atlandı ({filename}): {e}")
            continue
    
    return np.array(features_list), np.array(labels_list)

def train_model():
    """Modeli eğit"""
    print("=" * 50)
    print("SES TABANLI STRES ÖLÇÜM MODELİ EĞİTİMİ")
    print("=" * 50)
    
    # Veriyi yükle
    X, y = load_data('data')
    
    if X is None or len(X) == 0:
        print("HATA: Veri yüklenemedi!")
        return
    
    print(f"\nToplam {len(X)} örnek yüklendi.")
    print(f"Özellik sayısı: {X.shape[1]}")
    
    # Sınıf dağılımı
    unique, counts = np.unique(y, return_counts=True)
    print("\nSınıf dağılımı:")
    for label, count in zip(unique, counts):
        print(f"  {STRESS_LABELS[label]}: {count} örnek")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalizasyon
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nModel eğitiliyor...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Değerlendirme
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 50)
    print(f"Model Doğruluğu: {accuracy * 100:.2f}%")
    print("=" * 50)
    print("\nDetaylı Rapor:")
    print(classification_report(y_test, y_pred, target_names=[STRESS_LABELS[i] for i in sorted(STRESS_LABELS.keys())]))
    
    # Modeli kaydet
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stress_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✓ Model kaydedildi: models/stress_model.pkl")
    print("✓ Scaler kaydedildi: models/scaler.pkl")

if __name__ == "__main__":
    train_model()