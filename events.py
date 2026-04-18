"""
Canlı dinleyici ile panel arasındaki kayıtlar.

logs/events.jsonl  Yalnızca sona eklenen olay kaydı, satır başına bir JSON:
    detection    Tek bir 1 sn'lik pencerenin acil durum sınıfına düşmesi.
                 Tek başına alarm değildir.
    alarm        Operatör alarmı: bir bölümde (episode) art arda yeterli sayıda
                 tespit penceresi birikti. Her bölüm en fazla bir alarm üretir.
    episode_end  Ardışık tespitlerin bittiği an (normal konuşma ya da insan sesi yok).
logs/listener_status.json  Dinleyicinin her saniye yazdığı durum (heartbeat).
logs/listener.stop         Bu dosya oluşunca dinleyici kendini kapatır.

Panel log'u silmez, arşivler (os.replace). Dinleyici her olayda dosyayı açıp
kapattığı için arşivlemeden sonraki olay yeni bir dosyaya yazılır.
"""
import json
import os
from collections import Counter
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
EVENTS_FILE = os.path.join(LOG_DIR, "events.jsonl")
STATUS_FILE = os.path.join(LOG_DIR, "listener_status.json")
STOP_FILE = os.path.join(LOG_DIR, "listener.stop")
ARCHIVE_DIR = os.path.join(LOG_DIR, "archive")

# Bu kadar saniyedir heartbeat gelmeyen dinleyici "yanıt vermiyor" sayılır
HEARTBEAT_STALE_SEC = 5


def iso(t):
    return t.isoformat(timespec="milliseconds")


def append_event(event, path=EVENTS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_events(path=EVENTS_FILE):
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass    # yazılmakta olan son satır
    return events


def archive_events(path=EVENTS_FILE, archive_dir=ARCHIVE_DIR):
    """Log'u arşive taşır; taşınan dosyanın yolunu (yoksa None) döndürür."""
    if not os.path.exists(path):
        return None
    os.makedirs(archive_dir, exist_ok=True)
    dest = os.path.join(archive_dir, f"events_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    os.replace(path, dest)
    return dest


def write_status(status, path=STATUS_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False)
    try:
        os.replace(tmp, path)
    except PermissionError:
        pass    # Windows'ta panel o an okuyorsa; bir sonraki heartbeat yazar


def read_status(path=STATUS_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def listener_state(status, now=None):
    """'running' | 'unresponsive' | 'stopped' | 'error'"""
    if not status:
        return "stopped"
    if status.get("state") in ("stopped", "error"):
        return status["state"]
    now = now or datetime.now()
    age = (now - datetime.fromisoformat(status["last_seen"])).total_seconds()
    return "running" if age <= HEARTBEAT_STALE_SEC else "unresponsive"


class AlarmTracker:
    """Pencere sonuçlarını bölümlere ayırır ve alarm kararını verir.

    - Acil durum penceresi bir bölüm açar ya da açık bölümü sürdürür.
    - Normal konuşma veya insan sesi olmayan pencere bölümü kapatır.
    - Sessizlik bölümü kapatmaz (kişi nefes almak için susmuş olabilir).
    - Bölümde `consecutive` tespit birikince tek bir alarm üretilir;
      iki alarm arasında en az `cooldown_sec` saniye olur.
    """

    def __init__(self, consecutive=3, cooldown_sec=5):
        self.consecutive = consecutive
        self.cooldown_sec = cooldown_sec
        self.last_alarm = None
        self._n_episodes = 0
        self._reset()

    def _reset(self):
        self.episode = None
        self.episode_start = None
        self.windows = 0
        self.alarmed = False
        self.peak = 0.0
        self.states = Counter()

    def update(self, result, now):
        status = result.get("status")
        if status == "silence":
            return []

        if status == "DETECTED" and result.get("is_emergency"):
            if self.episode is None:
                self._n_episodes += 1
                self.episode = f"{now:%Y%m%d-%H%M%S}-{self._n_episodes}"
                self.episode_start = now
            self.windows += 1
            self.peak = max(self.peak, result["emergency_prob"])
            self.states[result["state"]] += 1
            events = [{
                "type": "detection",
                "time": iso(now),
                "episode": self.episode,
                "window": self.windows,
                "state": result["state"],
                "state_confidence": round(result["state_confidence"], 4),
                "human_prob": round(result["human_prob"], 4),
                "emergency_prob": round(result["emergency_prob"], 4),
            }]
            cooled = (self.last_alarm is None
                      or (now - self.last_alarm).total_seconds() >= self.cooldown_sec)
            if not self.alarmed and self.windows >= self.consecutive and cooled:
                self.alarmed = True
                self.last_alarm = now
                events.append({
                    "type": "alarm",
                    "time": iso(now),
                    "episode": self.episode,
                    "episode_start": iso(self.episode_start),
                    "windows": self.windows,
                    "states": dict(self.states),
                    "peak_emergency_prob": round(self.peak, 4),
                })
            return events

        if self.episode is None:
            return []
        event = {
            "type": "episode_end",
            "time": iso(now),
            "episode": self.episode,
            "windows": self.windows,
            "alarmed": self.alarmed,
        }
        self._reset()
        return [event]
