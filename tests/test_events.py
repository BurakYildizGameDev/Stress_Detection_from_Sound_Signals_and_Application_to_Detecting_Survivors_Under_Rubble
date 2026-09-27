import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from events import (AlarmTracker, append_event, archive_events, listener_state,
                    read_events)

T0 = datetime(2026, 1, 1, 12, 0, 0)

EMERGENCY = {"status": "DETECTED", "state": "panic", "is_emergency": True,
             "state_confidence": 0.7, "human_prob": 0.9, "emergency_prob": 0.8}
NORMAL = {"status": "DETECTED", "state": "normal", "is_emergency": False,
          "state_confidence": 0.8, "human_prob": 0.9, "emergency_prob": 0.2}
SILENCE = {"status": "silence"}
NO_HUMAN = {"status": "no_human", "human_prob": 0.05}


def run(tracker, results, start=T0):
    events = []
    for i, r in enumerate(results):
        events += tracker.update(r, start + timedelta(seconds=i))
    return events


def types(events):
    return [e["type"] for e in events]


def test_alarm_after_three_consecutive_windows():
    ev = run(AlarmTracker(3), [EMERGENCY] * 3)
    assert types(ev) == ["detection", "detection", "detection", "alarm"]
    assert ev[-1]["windows"] == 3


def test_long_episode_raises_a_single_alarm():
    ev = run(AlarmTracker(3), [EMERGENCY] * 10)
    assert types(ev).count("alarm") == 1
    assert types(ev).count("detection") == 10
    assert len({e["episode"] for e in ev}) == 1


def test_detections_below_threshold_are_not_alarms():
    ev = run(AlarmTracker(3), [EMERGENCY, EMERGENCY, NORMAL])
    assert types(ev) == ["detection", "detection", "episode_end"]
    assert ev[-1]["alarmed"] is False


def test_silence_does_not_break_an_episode():
    ev = run(AlarmTracker(3), [EMERGENCY, SILENCE, EMERGENCY, SILENCE, EMERGENCY])
    assert "alarm" in types(ev)


def test_no_human_closes_episode():
    ev = run(AlarmTracker(3), [EMERGENCY, EMERGENCY, NO_HUMAN, EMERGENCY])
    episodes = [e["episode"] for e in ev if e["type"] == "detection"]
    assert episodes[0] == episodes[1] != episodes[2]


def test_cooldown_between_episodes():
    t = AlarmTracker(3, cooldown_sec=60)
    ev = run(t, [EMERGENCY] * 3 + [NORMAL] + [EMERGENCY] * 3)
    assert types(ev).count("alarm") == 1
    ev = run(t, [NORMAL] + [EMERGENCY] * 3, start=T0 + timedelta(minutes=5))
    assert types(ev).count("alarm") == 1


def test_event_log_roundtrip_and_archive(tmp_path):
    log = tmp_path / "events.jsonl"
    append_event({"type": "detection", "n": 1}, str(log))
    append_event({"type": "alarm", "n": 2}, str(log))
    with open(log, "a", encoding="utf-8") as f:
        f.write('{"type": "detec')                   # yarım yazılmış satır
    assert [e["n"] for e in read_events(str(log))] == [1, 2]

    dest = archive_events(str(log), str(tmp_path / "archive"))
    assert not log.exists() and os.path.exists(dest)
    assert read_events(str(log)) == []
    append_event({"type": "detection", "n": 3}, str(log))   # arşivden sonra yeni dosya
    assert [e["n"] for e in read_events(str(log))] == [3]


def test_listener_state():
    now = T0
    assert listener_state(None, now) == "stopped"
    assert listener_state({"state": "running", "last_seen": T0.isoformat()}, now) == "running"
    old = (T0 - timedelta(seconds=30)).isoformat()
    assert listener_state({"state": "running", "last_seen": old}, now) == "unresponsive"
    assert listener_state({"state": "error", "last_seen": old}, now) == "error"
