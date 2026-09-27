"""
esp_serial_bridge.py - ESP32 firmware'inin seri porttan gönderdiği JSON satırlarını
panelin (app.py) okuduğu dosyalara yazar. live_listener.py yerine geçer: panel
mikrofonu bilgisayardan değil ESP'den dinliyormuş gibi çalışır.

  detection / alarm / episode_end  ->  logs/events.jsonl (events.py biçimi)
  status                           ->  logs/listener_status.json (heartbeat)
  boot / error                     ->  ekrana; error durumu panelde gösterilir

Cihazda gerçek saat yok; olay zamanı satırın bilgisayara ulaştığı andır.
Bölüm (episode) adları oturum ve cihaz açılışına göre önek alır, böylece
cihaz yeniden başladığında numaralar çakışmaz.

Kullanım:
    pip install pyserial
    python scripts/esp_serial_bridge.py --port COM3
    python scripts/esp_serial_bridge.py --replay firmware/test/sample_serial.txt
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import EVENTS_FILE, STATUS_FILE, STOP_FILE, append_event, iso, write_status

EVENT_TYPES = ("detection", "alarm", "episode_end")


class EspBridge:
    def __init__(self, device="ESP32-S3", events_path=EVENTS_FILE, status_path=STATUS_FILE,
                 now=None):
        self.device = device
        self.events_path = events_path
        self.status_path = status_path
        self.started_at = now or datetime.now()
        self.boot_id = f"{self.started_at:%Y%m%d-%H%M%S}"
        self.sample_rate = None
        self.error = None

    def episode_id(self, n):
        return f"esp-{self.boot_id}-{n}"

    def convert(self, msg, now):
        """Cihaz olayını events.py biçimine çevirir."""
        kind = msg["type"]
        out = {"type": kind, "time": iso(now), "episode": self.episode_id(msg["episode"])}
        if kind == "detection":
            for k in ("window", "state", "state_confidence", "human_prob", "emergency_prob"):
                out[k] = msg[k]
        elif kind == "alarm":
            start = now - timedelta(milliseconds=msg["t_ms"] - msg["episode_start_ms"])
            out.update({"episode_start": iso(start), "windows": msg["windows"],
                        "states": msg["states"], "peak_emergency_prob": msg["peak_emergency_prob"]})
        else:  # episode_end
            out.update({"windows": msg["windows"], "alarmed": msg["alarmed"]})
        return out

    def status(self, msg, now, state="running"):
        return {
            "state": state,
            "pid": os.getpid(),
            "started_at": iso(self.started_at),
            "last_seen": iso(now),
            "device": self.device,
            "sample_rate": self.sample_rate,
            "last_rms": msg.get("last_rms", 0.0),
            "last_status": msg.get("last_status"),
            "windows_processed": msg.get("windows", 0),
            "open_episode": self.episode_id(msg["open_episode"]) if msg.get("open_episode") else None,
            "inference_ms": msg.get("inference_ms"),
            "overruns": msg.get("overruns"),
            "error": self.error,
        }

    def handle_line(self, line, now=None):
        """Bir seri satırı işler; events.jsonl'e yazılan olayları döndürür.
        JSON olmayan satırlar (ROM açılış mesajları vb.) yok sayılır."""
        now = now or datetime.now()
        line = line.strip()
        if not line.startswith("{"):
            return []
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return []
        kind = msg.get("type")

        if kind in EVENT_TYPES:
            event = self.convert(msg, now)
            append_event(event, self.events_path)
            return [event]
        if kind == "status":
            write_status(self.status(msg, now), self.status_path)
        elif kind == "boot":
            # Cihaz yeniden başladı: bölüm numaraları 1'den başlar
            self.boot_id = f"{now:%Y%m%d-%H%M%S}"
            self.sample_rate = msg.get("sample_rate")
            self.error = None
            print(f"ESP açıldı: {msg}")
        elif kind == "error":
            self.error = msg.get("message", "bilinmeyen hata")
            write_status(self.status({}, now, state="error"), self.status_path)
            print(f"ESP hatası: {self.error}")
        return []

    def stopped(self):
        write_status(self.status({}, datetime.now(), state="stopped"), self.status_path)


def print_event(event):
    if event["type"] == "alarm":
        print(f"ALARM {event['time']} bölüm {event['episode']}: {event['windows']} pencere {event['states']}")
    elif event["type"] == "detection":
        print(f"tespit #{event['window']} {event['state']} ({event['emergency_prob']:.0%})")
    else:
        print(f"bölüm bitti {event['episode']} (alarm: {event['alarmed']})")


def read_serial(port, baud):
    try:
        import serial
    except ImportError:
        sys.exit("pyserial gerekli: pip install pyserial")
    with serial.Serial(port, baud, timeout=1) as ser:
        while not os.path.exists(STOP_FILE):
            raw = ser.readline()
            if raw:
                yield raw.decode("utf-8", errors="replace")


def read_replay(path, delay):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line
            if delay:
                time.sleep(delay)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--port", help="seri port (ör. COM3, /dev/ttyACM0)")
    src.add_argument("--replay", help="seri port yerine kaydedilmiş satırları oku")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--delay", type=float, default=0.0,
                   help="--replay'de satırlar arası bekleme (sn), panelde canlı izlemek için")
    args = p.parse_args()

    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)
    bridge = EspBridge(device=f"ESP32-S3 ({args.port or 'replay'})")
    lines = read_serial(args.port, args.baud) if args.port else read_replay(args.replay, args.delay)
    try:
        for line in lines:
            for event in bridge.handle_line(line):
                print_event(event)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stopped()
        print("Köprü durdu.")


if __name__ == "__main__":
    main()
