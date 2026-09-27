"""
firmware/lib/rubble_core (saf C) bilgisayarda derlenir ve Python karşılıklarıyla
karşılaştırılır:
  rubble_decision  <->  pipeline_v2.PipelineV2.analyze_audio_array (sahte modellerle)
  rubble_events    <->  events.AlarmTracker
Ayrıca C'nin ürettiği JSON satırları esp_serial_bridge'den geçirilir ve ESP
model başlık dosyalarının .pkl'lerden yeniden üretilebildiği denetlenir.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pipeline_v2
from events import AlarmTracker, listener_state, read_events, read_status
from scripts.esp_serial_bridge import EspBridge
from tests.test_c_model_export import CC, needs_cc

CORE_DIR = os.path.join(ROOT, "firmware", "lib", "rubble_core")
INCLUDE_DIR = os.path.join(ROOT, "firmware", "include")
EMERGENCY_CLASSES = ["moan", "normal", "panic", "stress", "whisper"]
T0 = datetime(2026, 1, 1, 12, 0, 0)

HARNESS = r"""
#include <stdio.h>
#include <string.h>
#include "rubble_decision.h"
#include "rubble_events.h"
#include "rubble_features.h"

static const char* const NAMES[5] = {"moan", "normal", "panic", "stress", "whisper"};

int main(void) {
    rubble_decision_config_t cfg = {0.0015f, 0.45f, 0.45f, 0, 1, 5};
    rubble_tracker_t tr;
    rubble_tracker_init(&tr, 3, 5000, 5);
    char cmd[4];
    while (scanf("%3s", cmd) == 1) {
        if (strcmp(cmd, "D") == 0) {          /* karar: rms hp[2] ep[5] */
            float rms, hp[2], ep[5];
            scanf("%f %f %f", &rms, &hp[0], &hp[1]);
            for (int i = 0; i < 5; i++) scanf("%f", &ep[i]);
            rubble_result_t r;
            if (!rubble_decide_silence(&cfg, rms, &r) && !rubble_decide_human(&cfg, rms, hp, &r))
                rubble_decide_emergency(&cfg, rms, hp[0], ep, &r);
            printf("%s %d %d %.9g %.9g %.9g\n", rubble_status_name(r.status), r.state,
                   r.is_emergency, r.human_prob, r.state_confidence, r.emergency_prob);
        } else if (strcmp(cmd, "T") == 0) {   /* takip: t_ms status state conf hp ep is_em */
            unsigned long t; int status;
            rubble_result_t r;
            scanf("%lu %d %d %f %f %f %d", &t, &status, &r.state, &r.state_confidence,
                  &r.human_prob, &r.emergency_prob, &r.is_emergency);
            r.status = (rubble_status_t)status;
            r.rms = 0.1f;
            rubble_event_t ev[RUBBLE_MAX_EVENTS];
            int n = rubble_tracker_update(&tr, &r, (uint32_t)t, ev);
            for (int i = 0; i < n; i++) {
                char buf[320];
                if (rubble_event_to_json(&ev[i], NAMES, 5, buf, sizeof(buf)) < 0) return 2;
                printf("%s\n", buf);
            }
            printf("END\n");
        } else if (strcmp(cmd, "R") == 0) {   /* rms: n x... */
            int n; static float x[4096];
            scanf("%d", &n);
            for (int i = 0; i < n; i++) scanf("%f", &x[i]);
            printf("%.9g\n", rubble_rms(x, n));
        } else if (strcmp(cmd, "J") == 0) {   /* küçük tamponda JSON taşması -1 olmalı */
            rubble_event_t e;
            memset(&e, 0, sizeof(e));
            e.type = RUBBLE_EVENT_DETECTION;
            char small[16];
            printf("%d\n", rubble_event_to_json(&e, NAMES, 5, small, sizeof(small)));
        }
    }
    return 0;
}
"""


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if CC is None:
        pytest.skip("C derleyicisi yok")
    tmp = tmp_path_factory.mktemp("fw")
    src = tmp / "harness.c"
    src.write_text(HARNESS, encoding="utf-8")
    exe = tmp / ("harness.exe" if os.name == "nt" else "harness")
    sources = [os.path.join(CORE_DIR, f) for f in sorted(os.listdir(CORE_DIR)) if f.endswith(".c")]
    subprocess.run(CC + ["-std=c99", "-O1", "-Wall", "-Wextra", "-Werror", "-I", CORE_DIR,
                         "-o", str(exe), str(src)] + sources + ["-lm"],
                   check=True, capture_output=True, text=True)

    def run(stdin):
        return subprocess.run([str(exe)], input=stdin, check=True,
                              capture_output=True, text=True).stdout
    return run


# ---------------------------------------------------------------- karar

class _FixedModel:
    def __init__(self, classes):
        self.classes_ = np.array(classes)
        self.probs = None

    def predict_proba(self, X):
        return np.array([self.probs])


def _decision_cases(n=400, seed=3):
    rng = np.random.default_rng(seed)
    cases = []
    while len(cases) < n:
        rms = float(rng.choice([0.0005, 0.0014, 0.0016, 0.05]))
        p_human = float(np.round(rng.choice([rng.uniform(), 0.40, 0.45, 0.50]), 3))
        ep = np.round(rng.dirichlet(np.ones(5) * rng.choice([0.3, 1.0, 5.0])), 3)
        if rng.uniform() < 0.1:
            ep = np.array([0.3, 0.3, 0.2, 0.1, 0.1])  # eşitlik: argmax ilk sınıfı seçmeli
        # float32 / float64 farkının karar değiştirebileceği tam sınır değerlerini atla
        if abs((1.0 - ep[1]) - 0.45) < 1e-6:
            continue
        cases.append((rms, p_human, ep.tolist()))
    return cases


def test_decision_matches_pipeline_v2(harness, monkeypatch):
    human, emerg = _FixedModel(["human", "non_human"]), _FixedModel(EMERGENCY_CLASSES)
    monkeypatch.setattr(pipeline_v2, "human_features_v2", lambda y, sr: np.zeros(33))
    monkeypatch.setattr(pipeline_v2, "emergency_features_v2", lambda y, sr: np.zeros(19))
    monkeypatch.setattr(pipeline_v2.librosa.feature, "rms", lambda y: np.array([[y[0]]]))
    pipe = pipeline_v2.PipelineV2(human_model=human, emergency_model=emerg, human_threshold=0.45)
    assert pipeline_v2.RMS_SILENCE_THRESHOLD == 0.0015
    assert pipeline_v2.EMERGENCY_PROB_THRESHOLD == 0.45

    cases = _decision_cases()
    stdin = "".join(f"D {rms} {ph} {1 - ph} " + " ".join(map(str, ep)) + "\n"
                    for rms, ph, ep in cases)
    lines = harness(stdin).strip().splitlines()
    assert len(lines) == len(cases)

    seen = set()
    for (rms, ph, ep), line in zip(cases, lines):
        human.probs, emerg.probs = [ph, 1 - ph], ep
        py = pipe.analyze_audio_array(np.full(8, rms, dtype=np.float32))
        status, state, is_em, hp, conf, eprob = line.split()
        seen.add(py["status"])
        assert status == py["status"], (rms, ph, ep)
        if py["status"] == "no_human":
            assert float(hp) == pytest.approx(py["human_prob"], abs=1e-6)
        if py["status"] == "DETECTED":
            assert EMERGENCY_CLASSES[int(state)] == py["state"], ep
            assert bool(int(is_em)) == py["is_emergency"], ep
            assert float(conf) == pytest.approx(py["state_confidence"], abs=1e-6)
            assert float(eprob) == pytest.approx(py["emergency_prob"], abs=1e-6)
    assert seen == {"silence", "no_human", "DETECTED"}


# ---------------------------------------------------------------- alarm takibi

STATUS_CODES = {"silence": 0, "no_human": 1, "DETECTED": 2}


def _random_results(rng, n):
    """(t_ms, result) dizisi; ara sıra uzun boşluklar bekleme süresini sınar."""
    t, out = 0, []
    for _ in range(n):
        t += int(rng.choice([1000, 1000, 1000, 7000]))
        kind = rng.choice(["silence", "no_human", "normal", "emergency", "emergency", "emergency"])
        if kind in ("silence", "no_human"):
            r = {"status": kind, "human_prob": 0.1}
        else:
            state = "normal" if kind == "normal" else str(rng.choice(["moan", "panic", "stress", "whisper"]))
            r = {"status": "DETECTED", "state": state, "is_emergency": kind == "emergency",
                 "state_confidence": round(float(rng.uniform(0.3, 1)), 4),
                 "human_prob": round(float(rng.uniform(0.45, 1)), 4),
                 "emergency_prob": round(float(rng.uniform(0.45, 1)), 4)}
        out.append((t, r))
    return out


def _tracker_stdin(seq):
    lines = []
    for t, r in seq:
        state = EMERGENCY_CLASSES.index(r["state"]) if "state" in r else -1
        lines.append(f"T {t} {STATUS_CODES[r['status']]} {state} {r.get('state_confidence', 0)} "
                     f"{r.get('human_prob', 0)} {r.get('emergency_prob', 0)} "
                     f"{int(bool(r.get('is_emergency')))}")
    return "\n".join(lines) + "\n"


def _parse_tracker_output(out):
    steps, cur = [], []
    for line in out.strip().splitlines():
        if line == "END":
            steps.append(cur)
            cur = []
        else:
            cur.append(json.loads(line))
    return steps


@pytest.mark.parametrize("seed", range(5))
def test_tracker_matches_alarm_tracker(harness, seed):
    seq = _random_results(np.random.default_rng(seed), 300)
    c_steps = _parse_tracker_output(harness(_tracker_stdin(seq)))
    assert len(c_steps) == len(seq)

    py_tracker = AlarmTracker(consecutive=3, cooldown_sec=5)
    episode_map = {}
    n_alarms = 0
    for (t, r), c_events in zip(seq, c_steps):
        py_events = py_tracker.update(r, T0 + timedelta(milliseconds=t))
        assert [e["type"] for e in c_events] == [e["type"] for e in py_events], (t, r)
        for c, p in zip(c_events, py_events):
            # bölüm eşleşmesi birebir olmalı (Python adı <-> C numarası)
            assert episode_map.setdefault(p["episode"], c["episode"]) == c["episode"]
            assert T0 + timedelta(milliseconds=c["t_ms"]) == datetime.fromisoformat(p["time"])
            if p["type"] == "detection":
                assert c["window"] == p["window"]
                assert c["state"] == p["state"]
                for k in ("state_confidence", "human_prob", "emergency_prob"):
                    assert c[k] == pytest.approx(p[k], abs=1e-4)
            elif p["type"] == "alarm":
                n_alarms += 1
                assert c["windows"] == p["windows"]
                assert c["states"] == p["states"]
                assert c["peak_emergency_prob"] == pytest.approx(p["peak_emergency_prob"], abs=1e-4)
                assert (T0 + timedelta(milliseconds=c["episode_start_ms"])
                        == datetime.fromisoformat(p["episode_start"]))
            else:
                assert c["windows"] == p["windows"]
                assert c["alarmed"] == p["alarmed"]
    assert len(set(episode_map.values())) == len(episode_map)
    assert n_alarms > 0


def test_tracker_cooldown_uses_unsigned_time(harness):
    # millis() ~49 günde taşar; bekleme süresi yine doğru hesaplanmalı
    W = 2**32
    em = {"status": "DETECTED", "state": "panic", "is_emergency": True,
          "state_confidence": 0.9, "human_prob": 0.9, "emergency_prob": 0.9}
    no_human = {"status": "no_human", "human_prob": 0.1}
    seq = [(W - 5000, em), (W - 4000, em), (W - 3000, em),   # alarm, W-3000'de
           (W - 2000, no_human),                             # bölüm biter
           (W - 1000, em), (0, em), (1000, em),              # 3. pencere: 4 sn geçti, alarm yok
           (2000, em)]                                       # 5 sn geçti: alarm
    steps = _parse_tracker_output(harness(_tracker_stdin(seq)))
    alarms = [t for (t, _), step in zip(seq, steps) if any(e["type"] == "alarm" for e in step)]
    assert alarms == [W - 3000, 2000]


def test_json_overflow_returns_error(harness):
    assert harness("J\n").strip() == "-1"


def test_rms_matches_numpy(harness):
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32) * 0.01
    out = harness(f"R {len(x)} " + " ".join(f"{v:.9g}" for v in x) + "\n")
    assert float(out) == pytest.approx(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), rel=1e-6)


# ---------------------------------------------------------------- köprü

def test_bridge_writes_dashboard_files_from_c_output(harness, tmp_path):
    seq = _random_results(np.random.default_rng(11), 60)
    c_lines = [json.dumps(e) for step in _parse_tracker_output(harness(_tracker_stdin(seq)))
               for e in step]
    events_path, status_path = str(tmp_path / "events.jsonl"), str(tmp_path / "status.json")
    bridge = EspBridge(events_path=events_path, status_path=status_path, now=T0)

    now = T0
    bridge.handle_line('{"type":"boot","sample_rate":22050}', now)
    bridge.handle_line("ESP-ROM:esp32s3-20210327", now)  # JSON olmayan satır yok sayılır
    written = []
    for line in c_lines:
        now += timedelta(seconds=1)
        written += bridge.handle_line(line, now)
    bridge.handle_line('{"type":"status","t_ms":5000,"windows":5,"last_rms":0.02,'
                       '"last_status":"DETECTED","open_episode":2}', now)

    events = read_events(events_path)
    assert events == written and len(events) == len(c_lines)
    for e in events:
        assert e["episode"].startswith(f"esp-{T0:%Y%m%d-%H%M%S}-")
        if e["type"] == "alarm":  # app.py'nin okuduğu alanlar
            assert {"time", "episode", "episode_start", "windows", "states",
                    "peak_emergency_prob"} <= set(e)
            assert datetime.fromisoformat(e["episode_start"]) <= datetime.fromisoformat(e["time"])
    status = read_status(status_path)
    assert listener_state(status, now=now) == "running"
    assert status["sample_rate"] == 22050 and status["open_episode"].endswith("-2")


def test_bridge_error_and_reboot(tmp_path):
    bridge = EspBridge(events_path=str(tmp_path / "e.jsonl"), status_path=str(tmp_path / "s.json"),
                       now=T0)
    bridge.handle_line('{"type":"error","message":"I2S başlatılamadı"}', T0)
    status = read_status(str(tmp_path / "s.json"))
    assert status["state"] == "error" and "I2S" in status["error"]

    later = T0 + timedelta(minutes=3)
    bridge.handle_line('{"type":"boot","sample_rate":22050}', later)
    ev = bridge.handle_line('{"type":"episode_end","t_ms":1,"episode":1,"windows":1,"alarmed":false}', later)
    assert ev[0]["episode"] == f"esp-{later:%Y%m%d-%H%M%S}-1"  # yeniden açılışta yeni önek


# ---------------------------------------------------------------- model başlıkları

@pytest.mark.parametrize("name,pkl", [("human_detector", "human_detector_esp"),
                                      ("emergency_classifier", "emergency_classifier_esp")])
def test_committed_model_headers_are_up_to_date(tmp_path, name, pkl):
    committed = os.path.join(INCLUDE_DIR, f"{name}_model.h")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "export_c_model.py"),
                    os.path.join(ROOT, "models", f"{pkl}.pkl"), "--name", name,
                    "--out", str(tmp_path)], check=True, capture_output=True)
    with open(committed, encoding="utf-8") as a, open(tmp_path / f"{name}_model.h", encoding="utf-8") as b:
        assert a.read() == b.read(), f"{committed} eski; export_c_model.py ile yeniden üretin"


@needs_cc
def test_headers_compile_as_cpp(tmp_path):
    # main.cpp C++ olarak derlenir; başlıkların C++ uyumlu olduğunu donanımsız sına
    src = tmp_path / "check.cpp"
    src.write_text('#include "config.h"\n#include "human_detector_model.h"\n'
                   '#include "emergency_classifier_model.h"\n#include "rubble_decision.h"\n'
                   '#include "rubble_events.h"\n#include "rubble_features.h"\n'
                   'static_assert(EMERGENCY_CLASSIFIER_NUM_CLASSES <= RUBBLE_MAX_CLASSES, "sınıf");\n'
                   'static_assert(HUMAN_DETECTOR_NUM_FEATURES == RUBBLE_HUMAN_N_FEATURES, "öznitelik");\n'
                   'static_assert(EMERGENCY_CLASSIFIER_NUM_FEATURES == RUBBLE_EMERGENCY_N_FEATURES, "öznitelik");\n'
                   'int main() { float f[33] = {0}; return human_detector_predict(f, nullptr); }\n',
                   encoding="utf-8")
    subprocess.run(CC + ["-x", "c++", "-std=c++11", "-c", "-o", str(tmp_path / "check.o"),
                         "-Wall", "-Werror", "-Wno-unused-function", "-Wno-unused-variable",
                         "-I", INCLUDE_DIR, "-I", CORE_DIR, str(src)],
                   check=True, capture_output=True, text=True)
