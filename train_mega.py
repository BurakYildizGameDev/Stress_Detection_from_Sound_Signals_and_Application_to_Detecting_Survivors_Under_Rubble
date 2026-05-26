import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Berlin EMODB
BERLIN_EMOTION = {
    'W': 2, 'L': 0, 'E': 1, 'A': 2, 'F': 0, 'T': 1, 'N': 0
}

# RAVDESS
RAVDESS_EMOTION = {
    '01': 0, '02': 0, '03': 0, '04': 1, '05': 2, '06': 2, '07': 1, '08': 1
}

# CREMA-D
CREMA_EMOTION = {
    'ANG': 2, 'DIS': 1, 'FEA': 2, 'HAP': 0, 'NEU': 0, 'SAD': 1
}

# TESS
TESS_EMOTION = {
    'angry': 2, 'disgust': 1, 'fear': 2, 'happy': 0, 
    'neutral': 0, 'pleasant_surprise': 0, 'ps': 0, 'sad': 1
}

# SAVEE
SAVEE_EMOTION = {
    'a': 2, 'd': 1, 'f': 2, 'h': 0, 'n': 0, 'sa': 1, 'su': 0
}

STRESS_LABELS = {0: 'Düşük Stres', 1: 'Orta Stres', 2: 'Yüksek Stres'}

def extract_features(file_path):
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
        return None

def load_berlin(data_dir):
    features, labels = [], []
    berlin_dir = os.path.join(data_dir, 'berlin')
    if not os.path.exists(berlin_dir):
        return [], []
    
    wav_files = [f for f in os.listdir(berlin_dir) if f.endswith('.wav')]
    print(f"\n[Berlin] {len(wav_files)} dosya bulundu")
    
    for i, f in enumerate(wav_files):
        try:
            emotion = f[5]
            if emotion not in BERLIN_EMOTION:
                continue
            feat = extract_features(os.path.join(berlin_dir, f))
            if feat is not None:
                features.append(feat)
                labels.append(BERLIN_EMOTION[emotion])
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(wav_files)} işlendi")
        except:
            continue
    
    print(f"[Berlin] {len(features)} örnek yüklendi")
    return features, labels

def load_ravdess(data_dir):
    features, labels = [], []
    ravdess_dir = os.path.join(data_dir, 'ravdess')
    if not os.path.exists(ravdess_dir):
        return [], []
    
    actors = [f for f in os.listdir(ravdess_dir) if f.startswith('Actor_')]
    print(f"\n[RAVDESS] {len(actors)} aktör bulundu")
    
    total = 0
    for actor in actors:
        actor_path = os.path.join(ravdess_dir, actor)
        for f in os.listdir(actor_path):
            if not f.endswith('.wav'):
                continue
            try:
                emotion = f.split('-')[2]
                if emotion not in RAVDESS_EMOTION:
                    continue
                feat = extract_features(os.path.join(actor_path, f))
                if feat is not None:
                    features.append(feat)
                    labels.append(RAVDESS_EMOTION[emotion])
                total += 1
                if total % 200 == 0:
                    print(f"  {total} dosya işlendi")
            except:
                continue
    
    print(f"[RAVDESS] {len(features)} örnek yüklendi")
    return features, labels

def load_savee(data_dir):
    features, labels = [], []
    savee_dir = os.path.join(data_dir, 'savee')
    if not os.path.exists(savee_dir):
        return [], []
    
    wav_files = [f for f in os.listdir(savee_dir) if f.endswith('.wav')]
    print(f"\n[SAVEE] {len(wav_files)} dosya bulundu")
    
    for i, f in enumerate(wav_files):
        try:
            # Dosya formatı: DC_a01.wav, JE_sa02.wav
            parts = f.replace('.wav', '').split('_')
            if len(parts) < 2:
                continue
            
            emotion = parts[1][0:2] if parts[1].startswith('sa') else parts[1][0]
            
            if emotion not in SAVEE_EMOTION:
                continue
            
            feat = extract_features(os.path.join(savee_dir, f))
            if feat is not None:
                features.append(feat)
                labels.append(SAVEE_EMOTION[emotion])
            
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(wav_files)} işlendi")
        except:
            continue
    
    print(f"[SAVEE] {len(features)} örnek yüklendi")
    return features, labels

def load_crema(data_dir):
    features, labels = [], []
    crema_dir = os.path.join(data_dir, 'crema')
    if not os.path.exists(crema_dir):
        return [], []
    
    wav_files = [f for f in os.listdir(crema_dir) if f.endswith('.wav')]
    print(f"\n[CREMA-D] {len(wav_files)} dosya bulundu")
    
    success_count = 0
    for i, f in enumerate(wav_files):
        try:
            # Dosya formatı: 1001_DFA_ANG_XX.wav
            parts = f.replace('.wav', '').split('_')
            if len(parts) < 3:
                continue
            
            emotion = parts[2]  # ANG, DIS, FEA, HAP, NEU, SAD
            if emotion not in CREMA_EMOTION:
                continue
            
            file_path = os.path.join(crema_dir, f)
            feat = extract_features(file_path)
            
            if feat is not None:
                features.append(feat)
                labels.append(CREMA_EMOTION[emotion])
                success_count += 1
            
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(wav_files)} işlendi ({success_count} başarılı)")
        except Exception as e:
            continue
    
    print(f"[CREMA-D] {len(features)} örnek yüklendi ({len(wav_files)} dosyadan)")
    return features, labels

def load_tess(data_dir):
    features, labels = [], []
    tess_dir = os.path.join(data_dir, 'tess')
    if not os.path.exists(tess_dir):
        return [], []
    
    folders = [f for f in os.listdir(tess_dir) if os.path.isdir(os.path.join(tess_dir, f))]
    print(f"\n[TESS] {len(folders)} klasör bulundu")
    
    total = 0
    for folder in folders:
        folder_path = os.path.join(tess_dir, folder)
        # Klasör adından duygu çıkar (örn: OAF_angry -> angry)
        emotion = folder.split('_')[1].lower() if '_' in folder else folder.lower()
        
        if emotion not in TESS_EMOTION:
            continue
        
        for f in os.listdir(folder_path):
            if not f.endswith('.wav'):
                continue
            try:
                feat = extract_features(os.path.join(folder_path, f))
                if feat is not None:
                    features.append(feat)
                    labels.append(TESS_EMOTION[emotion])
                total += 1
                if total % 200 == 0:
                    print(f"  {total} dosya işlendi")
            except:
                continue
    
    print(f"[TESS] {len(features)} örnek yüklendi")
    return features, labels

def train_model():
    print("=" * 70)
    print("MEGA MODEL EĞİTİMİ - TÜM VERİ SETLERİ")
    print("=" * 70)
    
    # Tüm veri setlerini yükle
    b_feat, b_lab = load_berlin('data')
    r_feat, r_lab = load_ravdess('data')
    c_feat, c_lab = load_crema('data')
    t_feat, t_lab = load_tess('data')
    s_feat, s_lab = load_savee('data')
    
    # Birleştir
    all_features = b_feat + r_feat + c_feat + t_feat + s_feat
    all_labels = b_lab + r_lab + c_lab + t_lab + s_lab
    
    if len(all_features) == 0:
        print("HATA: Hiç veri bulunamadı!")
        return
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'='*70}")
    print(f"TOPLAM: {len(X)} örnek")
    print(f"  - Berlin: {len(b_feat)}")
    print(f"  - RAVDESS: {len(r_feat)}")
    print(f"  - CREMA-D: {len(c_feat)}")
    print(f"  - TESS: {len(t_feat)}")
    print(f"  - SAVEE: {len(s_feat)}")
    print(f"Özellik sayısı: {X.shape[1]}")
    
    # Sınıf dağılımı
    unique, counts = np.unique(y, return_counts=True)
    print("\nSınıf Dağılımı:")
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
    
    print("\nModel eğitiliyor... (bu 5-10 dakika sürebilir)")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=2,
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
        target_names=[STRESS_LABELS[i] for i in sorted(STRESS_LABELS.keys())]
    ))
    
    # Model kaydet
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/stress_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✓ Model kaydedildi: models/stress_model.pkl")
    print("✓ Scaler kaydedildi: models/scaler.pkl")
    print(f"\n🚀 {len(X)} örnekle eğitildi! app.py ile kullanabilirsin!")

if __name__ == "__main__":
    train_model()