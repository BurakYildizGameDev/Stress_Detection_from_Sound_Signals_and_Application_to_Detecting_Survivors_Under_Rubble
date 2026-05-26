import os
import sys
import numpy as np
import librosa
import joblib
from audio_enhancement import AudioEnhancer
from voice_detector import VoiceActivityDetector
from scream_booster import ScreamDetector

# Duygu etiketleri
EMOTION_LABELS = {0: 'Düşük Stres', 1: 'Orta Stres', 2: 'Yüksek Stres'}

class StressPredictorEnhanced:
    """Geliştirilmiş stres tahmin sistemi"""
    
    def __init__(self, model_path='models/stress_model_ensemble.pkl', 
                 scaler_path='models/scaler_ultimate.pkl'):
        """Model yükle"""
        
        print("🔧 Model yükleniyor...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"  ✓ Model yüklendi")
        
        # Scaler varsa yükle
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Scaler yüklendi")
        else:
            self.scaler = None
            print(f"  ⚠️ Scaler bulunamadı (opsiyonel)")
        
        # Enhancer, VAD ve Scream Detector
        self.enhancer = AudioEnhancer()
        self.vad = VoiceActivityDetector(aggressiveness=2)
        self.scream_detector = ScreamDetector()
        
        print("✅ Sistem hazır!\n")
    
    def extract_features(self, y, sr):
        """Özellik çıkarma (57 özellik)"""
        try:
            # Energy
            rms = np.mean(librosa.feature.rms(y=y))
            rms_std = np.std(librosa.feature.rms(y=y))
            rms_max = np.max(librosa.feature.rms(y=y))
            
            # ZCR
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            zcr_std = np.std(librosa.feature.zero_crossing_rate(y))
            
            # Spectral
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
        except Exception as e:
            print(f"❌ Özellik çıkarma hatası: {e}")
            return None
    
    def predict(self, file_path, amplify_factor=3.5, check_voice=True):
        """Tahmin yap"""
        print("=" * 60)
        print("🎯 GELİŞTİRİLMİŞ STRES TAHMİNİ")
        print("=" * 60)
        print(f"📁 Dosya: {file_path}\n")
        
        # 1. Ses iyileştirme
        y_enhanced, sr = self.enhancer.enhance_full_pipeline(
            file_path,
            amplify_factor=amplify_factor,
            apply_noise_reduction=True,
            apply_bandpass=True,
            apply_compression=True
        )
        
        if y_enhanced is None:
            return None
        
        # 2. İnsan sesi kontrolü
        is_voice = True
        voice_conf = 1.0
        
        if check_voice:
            print("🎤 İnsan sesi kontrolü...")
            is_voice, voice_conf = self.vad.is_human_voice(y_enhanced, sr)
            
            if is_voice:
                print(f"  ✅ İnsan sesi tespit edildi (Güven: %{voice_conf*100:.1f})")
            else:
                print(f"  ⚠️ İnsan sesi tespit edilemedi (Güven: %{voice_conf*100:.1f})")
        
        # 2.5. Çığlık tespiti
        print("\n🔥 Çığlık/Acil durum kontrolü...")
        is_scream, scream_conf, scream_features = self.scream_detector.detect_scream_features(y_enhanced, sr)
        
        if is_scream:
            print(f"  🔴 ÇIĞLIK TESPİT EDİLDİ! (Güven: %{scream_conf*100:.1f})")
        else:
            print(f"  🟢 Normal ses (Güven: %{scream_conf*100:.1f})")
        
        # 3. Özellik çıkarma
        print("\n🔍 Özellikler çıkarılıyor...")
        features = self.extract_features(y_enhanced, sr)
        
        if features is None:
            return None
        
        print(f"  ✓ {len(features)} özellik çıkarıldı")
        
        # 4. Normalizasyon
        if self.scaler is not None:
            features = self.scaler.transform(features.reshape(1, -1))
        else:
            features = features.reshape(1, -1)
        
        # 5. Tahmin
        print("\n🤖 Model tahmini yapılıyor...")
        prediction = self.model.predict(features)[0]
        
        # Olasılıklar
        try:
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        except:
            confidence = 0.8
        
        # Çığlık düzeltmesi
        scream_adjusted = False
        if is_scream and scream_conf > 0.6:
            if prediction < 2:  # Düşük veya Orta stres
                print(f"  ⚡ ÇIĞLIK TESPİTİ: Tahmin düzeltiliyor...")
                print(f"     Orijinal: {EMOTION_LABELS[prediction]} (%{confidence*100:.1f})")
                prediction = 2  # Yüksek Stres'e çek
                confidence = max(confidence, scream_conf)  # Daha yüksek güveni al
                scream_adjusted = True
                print(f"     Düzeltilmiş: {EMOTION_LABELS[prediction]} (%{confidence*100:.1f})")
        
        label = EMOTION_LABELS[prediction]
        
        result = {
            'prediction': int(prediction),
            'label': label,
            'confidence': float(confidence),
            'is_human_voice': is_voice,
            'voice_confidence': float(voice_conf),
            'amplification_used': amplify_factor,
            'is_scream': is_scream,
            'scream_confidence': float(scream_conf),
            'scream_adjusted': scream_adjusted
        }
        
        # Sonuçlar
        print("\n" + "=" * 60)
        print("📊 SONUÇLAR")
        print("=" * 60)
        
        if prediction == 0:
            icon = "🟢"
        elif prediction == 1:
            icon = "🟡"
        else:
            icon = "🔴"
        
        print(f"{icon} Tahmin: {label}")
        print(f"📈 Güven: %{confidence*100:.1f}")
        print(f"🎤 İnsan sesi: {'✅ Evet' if is_voice else '❌ Hayır'} (%{voice_conf*100:.1f})")
        print(f"🔥 Çığlık tespiti: {'🔴 EVET' if is_scream else '🟢 Hayır'} (%{scream_conf*100:.1f})")
        print(f"🔊 Güçlendirme: {amplify_factor}x")
        
        if scream_adjusted:
            print(f"\n⚡ NOT: Çığlık tespiti nedeniyle tahmin YÜKSELTİLDİ!")
        
        if not is_voice and voice_conf < 0.3:
            print("\n⚠️ UYARI: İnsan sesi tespit edilemedi!")
        
        print("=" * 60)
        
        return result


def main():
    """Ana fonksiyon"""
    
    if len(sys.argv) < 2:
        print("Kullanım: python predict_enhanced.py <ses_dosyasi.wav> [güçlendirme]")
        print("\nÖrnek:")
        print("  python predict_enhanced.py temp.wav")
        print("  python predict_enhanced.py temp.wav 4.0")
        sys.exit(1)
    
    file_path = sys.argv[1]
    amplify_factor = float(sys.argv[2]) if len(sys.argv) > 2 else 3.5
    
    if not os.path.exists(file_path):
        print(f"❌ Dosya bulunamadı: {file_path}")
        sys.exit(1)
    
    try:
        predictor = StressPredictorEnhanced()
        result = predictor.predict(file_path, amplify_factor=amplify_factor)
        
        if result is None:
            print("❌ Tahmin başarısız!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ HATA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()