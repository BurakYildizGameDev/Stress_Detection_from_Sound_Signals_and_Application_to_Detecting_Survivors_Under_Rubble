import os
import sys
from datetime import datetime

import numpy as np

from events import (STOP_FILE, AlarmTracker, append_event, iso, write_status)
from pipeline import EMERGENCY_THRESHOLD, ModelLoadError, analyze_audio_array, get_model


class AudioMonitor:
    def __init__(self):
        # Ayarlar
        self.SAMPLE_RATE = 22050  # Pipeline ile uyumlu
        self.CHUNK_DURATION = 1.0
        self.CHUNK_SIZE = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        self.RMS_THRESHOLD = 0.01
        self.CONSECUTIVE_THRESHOLD = 3
        self.COOLDOWN = 5

        self.tracker = AlarmTracker(self.CONSECUTIVE_THRESHOLD, self.COOLDOWN)
        self.started_at = datetime.now()
        self.device = None
        self.last_rms = 0.0
        self.last_status = None
        self.n_windows = 0

    def status(self, state, error=None):
        return {
            "state": state,
            "pid": os.getpid(),
            "started_at": iso(self.started_at),
            "last_seen": iso(datetime.now()),
            "device": self.device,
            "sample_rate": self.SAMPLE_RATE,
            "last_rms": round(self.last_rms, 5),
            "last_status": self.last_status,
            "windows_processed": self.n_windows,
            "open_episode": self.tracker.episode,
            "error": error,
        }

    def trigger_emergency(self, alarm):
        print("\n" + "🚨" * 25)
        print("🚨🚨🚨 ACİL DURUM ALGILANDI! (ENKAZ ALARMI) 🚨🚨🚨")
        print(f"⏰ Zaman: {alarm['time']} | Bölüm: {alarm['episode']} | "
              f"{alarm['windows']} pencere | {alarm['states']}")
        print("🚨" * 25 + "\n")
        # Buraya SMS/Email/Telsiz entegrasyonu eklenebilir

    def rms(self, x):
        return float(np.sqrt(np.mean(x ** 2)))

    def process_audio(self, indata):
        # Flatten ve normalize
        audio_chunk = indata.flatten().astype(np.float32)
        self.n_windows += 1
        self.last_rms = self.rms(audio_chunk)

        try:
            # Sessizlik kontrolü (Erken çıkış)
            if self.last_rms < self.RMS_THRESHOLD:
                result = {"status": "silence", "rms": self.last_rms}
            else:
                result = analyze_audio_array(audio_chunk, sr=self.SAMPLE_RATE)
            self.last_status = result["status"]

            if result["status"] == "DETECTED":
                state = result["state"]
                icon = "🟢" if state == "normal" else "🔴"
                print(f"{icon} {state.upper():10} | "
                      f"Güven: {result['state_confidence']:.1%} | "
                      f"İnsan: {result['human_prob']:.1%} | "
                      f"Acil Durum: {result['emergency_prob']:.1%}")

            for event in self.tracker.update(result, datetime.now()):
                append_event(event)
                if event["type"] == "detection":
                    print(f"⚠️  Tespit penceresi #{event['window']} "
                          f"(alarm eşiği {self.CONSECUTIVE_THRESHOLD}, bölüm {event['episode']})")
                elif event["type"] == "alarm":
                    self.trigger_emergency(event)
                elif event["type"] == "episode_end":
                    print("✅ Normal duruma dönüldü.")

        except Exception as e:
            print(f"❌ İşleme Hatası: {e}")

    def start(self):
        print("=" * 60)
        print("🚨 ENKAZ ALTI AKUSTİK ACİL DURUM DİNLEYİCİSİ")
        print("=" * 60)

        try:
            get_model("human")
            get_model("emergency")
        except ModelLoadError as e:
            print(f"❌ Model yüklenemedi:\n{e}")
            write_status(self.status("error", str(e)))
            return 1

        import sounddevice as sd
        try:
            self.device = sd.query_devices(kind="input")["name"]
        except Exception as e:
            print(f"❌ Mikrofon bulunamadı: {e}")
            write_status(self.status("error", f"Mikrofon bulunamadı: {e}"))
            return 1

        if os.path.exists(STOP_FILE):
            os.remove(STOP_FILE)

        print(f"📊 Örnekleme Hızı: {self.SAMPLE_RATE} Hz")
        print(f"📊 Acil Durum Eşiği: {EMERGENCY_THRESHOLD:.2f}")
        print(f"📊 Ardışık Doğrulama Sayacı: {self.CONSECUTIVE_THRESHOLD}")
        print(f"📊 Mikrofon: {self.device}")
        print("🎧 Canlı dinleme başladı... (Durdurmak için Ctrl+C)\n")
        write_status(self.status("running"))

        try:
            with sd.InputStream(
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=self.CHUNK_SIZE,
                latency='low'
            ) as stream:
                while not os.path.exists(STOP_FILE):
                    data, overflowed = stream.read(self.CHUNK_SIZE)
                    if overflowed:
                        print("⚠️ Audio buffer overflow")
                    self.process_audio(data)
                    write_status(self.status("running"))
            print("\n⛔ Panelden durduruldu.")

        except KeyboardInterrupt:
            print("\n⛔ Dinleme durduruldu.")
        except Exception as e:
            print(f"❌ Kritik Donanım/Ses Hatası: {e}")
            write_status(self.status("error", f"Ses hatası: {e}"))
            return 1
        finally:
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)

        write_status(self.status("stopped"))
        return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(AudioMonitor().start())
