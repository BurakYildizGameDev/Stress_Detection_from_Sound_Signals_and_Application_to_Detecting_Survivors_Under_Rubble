import os
import json
import time
from datetime import datetime
import numpy as np
import sounddevice as sd
from pipeline import analyze_audio_array

class AudioMonitor:
    def __init__(self):
        # Ayarlar
        self.SAMPLE_RATE = 22050  # Pipeline ile uyumlu
        self.CHUNK_DURATION = 1.0
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        self.RMS_THRESHOLD = 0.01
        self.EMERGENCY_THRESHOLD = 0.50
        self.CONSECUTIVE_THRESHOLD = 3
        self.COOLDOWN = 5
        
        # Log dosya yolu
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_file = os.path.join(self.base_dir, "alerts_log.json")

        # Durum Değişkenleri
        self.emergency_count = 0
        self.last_alert_time = None
        self.alerts_log = []
        
        # Logları yükle (varsa)
        self.load_logs()

    def load_logs(self):
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r", encoding="utf-8") as f:
                    self.alerts_log = json.load(f)
            else:
                self.alerts_log = []
        except (FileNotFoundError, json.JSONDecodeError):
            self.alerts_log = []

    def save_logs(self):
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.alerts_log, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Log kaydetme hatası: {e}")

    def log_alert(self, state, human_prob, emergency_prob):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = {
            "timestamp": timestamp,
            "state": state,
            "human_prob": round(human_prob, 4),
            "emergency_prob": round(emergency_prob, 4)
        }
        self.alerts_log.append(alert)
        self.save_logs()

    def trigger_emergency(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print("\n" + "🚨" * 25)
        print("🚨🚨🚨 ACİL DURUM ALGILANDI! (ENKAZ ALARMI) 🚨🚨🚨")
        print(f"⏰ Zaman: {timestamp}")
        print("🚨" * 25 + "\n")
        # Buraya SMS/Email/Telsiz entegrasyonu eklenebilir

    def rms(self, x):
        return float(np.sqrt(np.mean(x ** 2)))

    def process_audio(self, indata):
        # Flatten ve normalize
        audio_chunk = indata.flatten().astype(np.float32)
        
        # Sessizlik kontrolü (Erken çıkış)
        if self.rms(audio_chunk) < self.RMS_THRESHOLD:
            return

        try:
            # Analiz
            result = analyze_audio_array(audio_chunk, sr=self.SAMPLE_RATE)
            status = result.get("status")

            if status == "DETECTED":
                state = result.get('state', 'normal')
                human_prob = result.get('human_prob', 0.0)
                emergency_prob = result.get('emergency_prob', 0.0)
                is_emergency = result.get('is_emergency', False)
                state_confidence = result.get('state_confidence', 0.0)

                icon = "🟢" if state == "normal" else "🔴"
                print(f"{icon} {state.upper():10} | "
                      f"Güven: {state_confidence:.1%} | "
                      f"İnsan: {human_prob:.1%} | "
                      f"Acil Durum: {emergency_prob:.1%}")
                
                # Sadece gerçek bir acil durum sınıfı varsa ve eşik aşılmışsa sayaç artsın
                if is_emergency or (state != "normal" and emergency_prob >= self.EMERGENCY_THRESHOLD):
                    self.emergency_count += 1
                    self.log_alert(state, human_prob, emergency_prob)
                    print(f"⚠️  ACİL DURUM SAYACI: {self.emergency_count}/{self.CONSECUTIVE_THRESHOLD}")
                    
                    if self.emergency_count >= self.CONSECUTIVE_THRESHOLD:
                        now = datetime.now()
                        if (self.last_alert_time is None or 
                            (now - self.last_alert_time).total_seconds() > self.COOLDOWN):
                            self.trigger_emergency()
                            self.last_alert_time = now
                        self.emergency_count = 0
                else:
                    if self.emergency_count > 0:
                        self.emergency_count = 0
                        print("✅ Normal duruma dönüldü.")
            
            elif status == "no_human":
                if self.emergency_count > 0:
                    self.emergency_count = 0

        except Exception as e:
            print(f"❌ İşleme Hatası: {e}")

    def start(self):
        print("=" * 60)
        print("🚨 ENKAZ ALTI AKUSTİK ACİL DURUM DİNLEYİCİSİ")
        print("=" * 60)
        print(f"📊 Örnekleme Hızı: {self.SAMPLE_RATE} Hz")
        print(f"📊 Acil Durum Eşiği: {self.EMERGENCY_THRESHOLD:.2f}")
        print(f"📊 Ardışık Doğrulama Sayacı: {self.CONSECUTIVE_THRESHOLD}")
        print(f"📊 Mikrofon: Varsayılan Giriş Aygıtı")
        print("🎧 Canlı dinleme başladı... (Durdurmak için Ctrl+C)\n")

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.CHUNK_SIZE,
                latency='low'
            ) as stream:
                while True:
                    data, overflowed = stream.read(self.CHUNK_SIZE)
                    if overflowed:
                        print("⚠️ Audio buffer overflow")
                    self.process_audio(data)
                    
        except KeyboardInterrupt:
            print("\n⛔ Dinleme durduruldu.")
        except Exception as e:
            print(f"❌ Kritik Donanım/Ses Hatası: {e}")

if __name__ == "__main__":
    monitor = AudioMonitor()
    monitor.start()