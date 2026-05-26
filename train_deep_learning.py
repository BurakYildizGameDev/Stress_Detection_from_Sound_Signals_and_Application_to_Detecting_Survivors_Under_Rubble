import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Kullanılabilir mi: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Duygu etiketleri
EMOTION_LABELS = {0: 'Düşük Stres', 1: 'Orta Stres', 2: 'Yüksek Stres'}

def extract_mel_spectrogram(file_path, n_mels=128, max_len=130):
    """Mel-spektrogram çıkar (CNN için)"""
    try:
        y, sr = librosa.load(file_path, duration=3, sr=22050)
        
        # Mel-spektrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Sabit uzunluğa getir
        if mel_spec_db.shape[1] < max_len:
            pad_width = max_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_len]
        
        return mel_spec_db
    except:
        return None

def get_emotion_from_filename(filename, dataset_name):
    """Dosya adından duygu etiketini çıkar"""
    try:
        if dataset_name == 'berlin':
            emotion = filename[5]
            mapping = {'W': 2, 'L': 0, 'E': 1, 'A': 2, 'F': 0, 'T': 1, 'N': 0}
            
        elif dataset_name == 'ravdess':
            parts = filename.split('-')
            emotion = parts[2] if len(parts) > 2 else None
            mapping = {'01': 0, '02': 0, '03': 0, '04': 1, '05': 2, '06': 2, '07': 1, '08': 1}
            
        elif dataset_name == 'tess':
            if 'angry' in filename or 'fear' in filename:
                return 2
            elif 'sad' in filename or 'disgust' in filename:
                return 1
            else:
                return 0
                
        elif dataset_name == 'savee':
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

def load_data_for_cnn(data_dir, n_mels=128, max_len=130):
    """CNN için veri yükle"""
    features_list = []
    labels_list = []
    
    datasets = ['berlin', 'ravdess', 'tess', 'savee']
    
    print("=" * 70)
    print("MEL-SPEKTROGRAM ÇIKARMA (CNN İÇİN)")
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
                mel_spec = extract_mel_spectrogram(file_path, n_mels, max_len)
                
                if mel_spec is not None:
                    features_list.append(mel_spec)
                    labels_list.append(label)
                
                if (i + 1) % 1000 == 0:
                    print(f"  {i + 1}/{len(wav_files)} işlendi...")
            except:
                continue
        
        print(f"[{dataset.upper()}] Tamamlandı")
    
    return np.array(features_list), np.array(labels_list)

def create_cnn_lstm_model(input_shape, num_classes=3):
    """CNN + LSTM Hibrit Model"""
    model = keras.Sequential([
        # CNN Katmanları - Spektrogram'dan özellik çıkar
        layers.Input(shape=input_shape),
        layers.Reshape((*input_shape, 1)),  # (128, 130) -> (128, 130, 1)
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Zaman serisi için reshape
        layers.Reshape((-1, 128)),  # LSTM için hazırla
        
        # LSTM Katmanları - Temporal patterns
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        
        # Dense Katmanları
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        # Output
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history, save_path='models/training_history.png'):
    """Eğitim geçmişini görselleştir"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Eğitim grafiği kaydedildi: {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path='models/confusion_matrix.png'):
    """Confusion matrix görselleştir"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS.values(),
                yticklabels=EMOTION_LABELS.values())
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix kaydedildi: {save_path}")

def train_model():
    """Model eğit"""
    print("\n" + "=" * 70)
    print("DERİN ÖĞRENME MODELİ EĞİTİMİ (CNN + LSTM)")
    print("=" * 70)
    
    # Veriyi yükle
    X, y = load_data_for_cnn('data_augmented')
    
    if len(X) == 0:
        print("HATA: Veri bulunamadı!")
        return
    
    print(f"\nToplam örnek: {len(X)}")
    print(f"Spektrogram boyutu: {X.shape[1:]} (n_mels x time)")
    
    # Sınıf dağılımı
    unique, counts = np.unique(y, return_counts=True)
    print("\nSınıf Dağılımı:")
    for label, count in zip(unique, counts):
        print(f"  {EMOTION_LABELS[label]}: {count} (%{count/len(y)*100:.1f})")
    
    # Train-val-test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Normalizasyon
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(X_test.shape)
    
    # Model oluştur
    print("\nModel oluşturuluyor...")
    model = create_cnn_lstm_model(input_shape=X_train.shape[1:], num_classes=3)
    
    # Model özeti
    model.summary()
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
    ]
    
    # Eğitim
    print("\nModel eğitiliyor... (Bu 15-20 dakika sürebilir)")
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Test
    y_pred = np.argmax(model.predict(X_test_scaled), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print(f"🎉 MODEL DOĞRULUĞU: {accuracy * 100:.2f}% 🎉")
    print("=" * 70)
    print("\nDetaylı Rapor:")
    print(classification_report(
        y_test, y_pred,
        target_names=[EMOTION_LABELS[i] for i in sorted(EMOTION_LABELS.keys())]
    ))
    
    # Görseller
    os.makedirs('models', exist_ok=True)
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    # Modeli kaydet
    model.save('models/stress_model_cnn_lstm.h5')
    joblib.dump(scaler, 'models/scaler_cnn_lstm.pkl')
    
    print(f"\n✓ Model kaydedildi: models/stress_model_cnn_lstm.h5")
    print(f"✓ Scaler kaydedildi: models/scaler_cnn_lstm.pkl")
    print(f"✓ {len(X)} örnek ile eğitildi")
    print(f"✓ CNN + LSTM hibrit mimari kullanıldı")

if __name__ == "__main__":
    train_model()