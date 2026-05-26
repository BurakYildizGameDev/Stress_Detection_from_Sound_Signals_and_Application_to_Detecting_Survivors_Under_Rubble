import sounddevice as sd
import numpy as np
from pipeline import analyze_audio_array
from datetime import datetime
import json
import time

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
        
        # Durum Değişkenleri
        self.emergency_count = 0
        self.last_alert_time = None
        self.alerts_log = []
        
        # Logları yükle (varsa)
        self.load_logs()

    def load_logs(self):
        try:
            with open("alerts_log.json", "r") as f:
                self.alerts_log = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.alerts_log = []

    def save_logs(self):
        try:
            with open("alerts_log.json", "w") as f:
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
        print(f"🚨🚨🚨 ACİL DURUM ALGILANDI! 🚨🚨🚨")
        print(f"⏰ Zaman: {timestamp}")
        print("🚨" * 25 + "\n")
        # Buraya SMS/Email entegrasyonu eklenebilir

    def rms(self, x):
        return np.sqrt(np.mean(x ** 2))

    def process_audio(self, indata):
        # Flatten ve normalize
        audio_chunk = indata.flatten().astype(np.float32)
        
        # Sessizlik kontrolü (Erken çıkış)
        if self.rms(audio_chunk) < self.RMS_THRESHOLD:
            return

        try:
            # Analiz (Resample gerekmez çünkü SR uyumlu)
            result = analyze_audio_array(audio_chunk)
            
            if result.get("status") == "DETECTED":
                state = result.get('state', 'Bilinmiyor')
                human_prob = result.get('human_prob', 0)
                emergency_prob = result.get('emergency_prob', 0)
                
                print(f"\n👤 {state.upper():10} | "
                      f"Human: {human_prob:.1%} | "
                      f"Emergency: {emergency_prob:.1%}")
                
                if emergency_prob > self.EMERGENCY_THRESHOLD:
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
                             self.emergency_count = 0
                else:
                    if self.emergency_count > 0:
                        self.emergency_count = 0
                        print(f"✅ Normal duruma döndü.")
            
            elif result.get("status") == "no_human":
                self.emergency_count = 0

        except Exception as e:
            print(f"❌ İşleme Hatası: {e}")

    def start(self):
        print(f"📊 Konfigürasyon:")
        print(f"   SR: {self.SAMPLE_RATE}")
        print(f"   Emergency Threshold: {self.EMERGENCY_THRESHOLD}")
        print(f"   Device: Default Microphone")
        print("\n🎧 Canlı dinleme başladı... (Ctrl+C ile durdur)\n")

        # Buffer yönetimi
        buffer = np.zeros(0, dtype=np.float32)

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
            print(f"❌ Kritik Hata: {e}")

if __name__ == "__main__":
    monitor = AudioMonitor()
    monitor.start()