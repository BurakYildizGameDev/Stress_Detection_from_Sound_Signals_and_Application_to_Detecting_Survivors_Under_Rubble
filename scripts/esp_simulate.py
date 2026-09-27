"""
esp_simulate.py - ESP32 firmware'inin karar hattını bilgisayarda, gerçek ses
dosyaları üzerinde çalıştırır (donanım gerekmez).

    ses dosyası -> 1 sn pencereler -> öznitelikler (Python/librosa)
                -> firmware/sim/sim_main.c: cihazdaki C kodu (RMS, modeller,
                   karar, alarm takibi) -> JSON olaylar

Her pencerenin sonucu aynı ESP modelleriyle çalışan Python hattıyla
(pipeline_v2.PipelineV2) karşılaştırılır. Öznitelik çıkarımı henüz C'de yok
(ESP-4); bu yüzden simülasyon öznitelik adımı dışındaki her şeyi sınar.

Seçenekler:
  --dashboard        olayları esp_serial_bridge üzerinden panelin dosyalarına yaz
                     (streamlit run app.py ile izlenir; --delay ile gerçek zamanlı)
  --export-scenario  Wokwi senaryosu üret (firmware/include/sim_scenario.h):
                     pencerelerin öznitelikleri + C'nin beklenen sonuçları
  --report           özet JSON (ör. reports/esp_simulation.json)
  --serial-out       cihazın seri çıktısı biçiminde satırlar (esp_serial_bridge --replay için)

Kullanım:
    python scripts/esp_simulate.py samples/woman_scream.mp3
    python scripts/esp_simulate.py samples/*.mp3 --report reports/esp_simulation.json
    python scripts/esp_simulate.py samples/people_talk.mp3 --dashboard --delay 1
C derleyicisi gerekir: gcc / clang, ya da Windows'ta `pip install ziglang`.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features import HUMAN_SR
from features_v2 import emergency_features_v2, human_features_v2
from pipeline_v2 import PipelineV2
from scripts.c_compiler import find_c_compiler
from scripts.export_c_model import c_float

FIRMWARE = os.path.join(ROOT, "firmware")
INCLUDE_DIR = os.path.join(FIRMWARE, "include")
CORE_DIR = os.path.join(FIRMWARE, "lib", "rubble_core")
SIM_SRC = os.path.join(FIRMWARE, "sim", "sim_main.c")
BUILD_DIR = os.path.join(FIRMWARE, "sim", "build")
SCENARIO_PATH = os.path.join(INCLUDE_DIR, "sim_scenario.h")
MODELS = {"human": os.path.join(ROOT, "models", "human_detector_esp.pkl"),
          "emergency": os.path.join(ROOT, "models", "emergency_classifier_esp.pkl")}

SR = HUMAN_SR            # cihaz 22050 Hz'de okur
WINDOW = SR              # 1 sn
STATUS_CODES = {"silence": 0, "no_human": 1, "DETECTED": 2, "unclassified": 3}


# ------------------------------------------------------------------ derleme

def _sources():
    files = [SIM_SRC] + [os.path.join(CORE_DIR, f) for f in os.listdir(CORE_DIR)]
    files += [os.path.join(INCLUDE_DIR, f) for f in ("rubble_pipeline.h", "human_detector_model.h",
                                                   "emergency_classifier_model.h")]
    return files


def build_simulator(cc=None, force=False):
    """firmware/sim/build altında derler; kaynaklar değişmediyse yeniden derlemez."""
    exe = os.path.join(BUILD_DIR, "esp_sim.exe" if os.name == "nt" else "esp_sim")
    if not force and os.path.exists(exe) and \
            os.path.getmtime(exe) >= max(os.path.getmtime(f) for f in _sources()):
        return exe
    cc = cc or find_c_compiler()
    if cc is None:
        sys.exit("C derleyicisi bulunamadı: gcc/clang kurun ya da `pip install ziglang`.")
    os.makedirs(BUILD_DIR, exist_ok=True)
    c_files = [SIM_SRC] + [os.path.join(CORE_DIR, f) for f in sorted(os.listdir(CORE_DIR))
                           if f.endswith(".c")]
    print("Simülatör derleniyor (modeller dahil, ~10-30 sn)...", flush=True)
    subprocess.run(cc + ["-std=c99", "-O1", "-I", INCLUDE_DIR, "-I", CORE_DIR, "-o", exe]
                   + c_files + ["-lm"], check=True)
    return exe


# ------------------------------------------------------------------ ses

def load_windows(paths, gap_sec=1.0):
    """Dosyaları arka arkaya (araya sessizlik koyarak) 1 sn'lik pencerelere böler.
    Son yarım pencere atılır (canlı dinleyici gibi yalnızca tam pencereler)."""
    windows, sources = [], []
    for i, path in enumerate(paths):
        y, _ = librosa.load(path, sr=SR)
        if i > 0 and gap_sec > 0:
            for _ in range(int(round(gap_sec))):
                windows.append(np.zeros(WINDOW, dtype=np.float32))
                sources.append("(sessizlik)")
        for k in range(len(y) // WINDOW):
            windows.append(y[k * WINDOW:(k + 1) * WINDOW].astype(np.float32))
            sources.append(os.path.basename(path))
    return windows, sources


def window_features(y):
    return human_features_v2(y, sr=SR), emergency_features_v2(y, sr=SR)


# ------------------------------------------------------------------ çalıştırma

def run_c(exe, windows, features):
    lines = []
    for y, (h, e) in zip(windows, features):
        lines.append("W %d %s 1 %s %s" % (len(y), " ".join(f"{v:.9g}" for v in y),
                                          " ".join(f"{v:.9g}" for v in h),
                                          " ".join(f"{v:.9g}" for v in e)))
    out = subprocess.run([exe], input="\n".join(lines) + "\n", capture_output=True,
                         text=True, check=True).stdout
    results, events = [], []
    for line in out.strip().splitlines():
        msg = json.loads(line)
        if msg["type"] == "window":
            results.append(msg)
            events.append([])
        else:
            events[-1].append(msg)
    return results, events


def run_python(windows):
    """Aynı ESP modelleriyle Python hattı (referans)."""
    human, emergency = joblib.load(MODELS["human"]), joblib.load(MODELS["emergency"])
    with open(os.path.splitext(MODELS["human"])[0] + ".json", encoding="utf-8") as f:
        threshold = json.load(f)["human_threshold"]
    pipe = PipelineV2(human_model=human, emergency_model=emergency, human_threshold=threshold)
    return [pipe.analyze_audio_array(y, sr=SR) for y in windows]


def compare(c_results, py_results):
    mismatches = []
    for i, (c, p) in enumerate(zip(c_results, py_results)):
        same = c["status"] == p["status"]
        if same and p["status"] == "DETECTED":
            same = c["state"] == p["state"] and c["is_emergency"] == p["is_emergency"]
        if not same:
            mismatches.append({"window": i + 1, "c": [c["status"], c["state"]],
                               "python": [p["status"], p.get("state", "-")]})
    return mismatches


# ------------------------------------------------------------------ çıktılar

def print_table(results, events, sources):
    print(f"\n{'sn':>4}  {'kaynak':18s} {'durum':10s} {'sınıf':8s} {'insan':>6s} {'acil':>6s}  olay")
    for r, evs, src in zip(results, events, sources):
        ev = ", ".join(e["type"] for e in evs)
        hp = f"{r['human_prob']:.2f}" if r["status"] in ("no_human", "DETECTED") else ""
        ep = f"{r['emergency_prob']:.2f}" if r["status"] == "DETECTED" else ""
        print(f"{r['window']:4d}  {src[:18]:18s} {r['status']:10s} {r['state']:8s} {hp:>6s} {ep:>6s}  {ev}")


def serial_lines(results, events):
    """Firmware'in seri çıktısı biçiminde satırlar: boot, olaylar, pencere başına status."""
    lines = [json.dumps({"type": "boot", "mode": "simulator", "sample_rate": SR,
                         "features": "python"}, separators=(",", ":"))]
    open_episode = 0
    for r, evs in zip(results, events):
        for e in evs:
            lines.append(json.dumps(e, separators=(",", ":")))
            open_episode = e["episode"] if e["type"] == "detection" else (
                0 if e["type"] == "episode_end" else open_episode)
        lines.append(json.dumps({"type": "status", "t_ms": r["t_ms"] + 1000, "windows": r["window"],
                                 "last_rms": round(r["rms"], 5), "last_status": r["status"],
                                 "open_episode": open_episode}, separators=(",", ":")))
    return lines


def to_dashboard(results, events, delay):
    from scripts.esp_serial_bridge import EspBridge
    bridge = EspBridge(device="ESP32-S3 (simülatör)")
    for line in serial_lines(results, events):
        bridge.handle_line(line)
        if delay and '"type":"status"' in line:
            time.sleep(delay)
    bridge.stopped()


def export_scenario(path, results, features, sources, source_desc):
    n_h, n_e = len(features[0][0]), len(features[0][1])
    rows = []
    for r, (h, e), src in zip(results, features, sources):
        state = -1 if r["state"] == "-" else int(_class_index(r["state"]))
        rows.append("    {%s, 1,\n     {%s},\n     {%s},\n     %d, %d},  /* %d: %s, %s %s */" % (
            c_float(r["rms"]), ", ".join(c_float(v) for v in h), ", ".join(c_float(v) for v in e),
            STATUS_CODES[r["status"]], state, r["window"], src, r["status"], r["state"]))
    code = f"""/*
 * sim_scenario.h - Wokwi senaryosu. Otomatik üretildi, elle düzenlemeyin:
 *   python scripts/esp_simulate.py {source_desc} --export-scenario
 *
 * Wokwi'de mikrofon ve I2S simüle edilmediği için firmware (RUBBLE_SIM_SCENARIO)
 * bu pencereleri saniyede bir oynatır. Öznitelikler Python'da (librosa) gerçek
 * kayıtlardan çıkarıldı; expected_* değerleri aynı C kodunun bilgisayardaki
 * sonucudur (scripts/esp_simulate.py). Cihaz her pencerede bunları karşılaştırır.
 */
#ifndef SIM_SCENARIO_H
#define SIM_SCENARIO_H

#define SIM_SCENARIO_WINDOWS {len(rows)}

typedef struct {{
    float rms;
    int has_features;
    float human[{n_h}];
    float emergency[{n_e}];
    int expected_status;  /* rubble_status_t */
    int expected_state;   /* acil durum sınıf indeksi, yoksa -1 */
}} sim_window_t;

static const sim_window_t SIM_SCENARIO[SIM_SCENARIO_WINDOWS] = {{
{chr(10).join(rows)}
}};

#endif /* SIM_SCENARIO_H */
"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(code)
    print(f"-> {os.path.relpath(path, ROOT)} ({len(rows)} pencere)")


def _class_index(name):
    with open(os.path.splitext(MODELS["emergency"])[0] + ".json", encoding="utf-8") as f:
        return json.load(f)["labels"].index(name)


def simulate(paths, gap_sec=1.0, cc=None):
    windows, sources = load_windows(paths, gap_sec)
    if not windows:
        sys.exit("Dosyalarda 1 sn'lik tam pencere yok.")
    features = [window_features(y) for y in windows]
    exe = build_simulator(cc)
    results, events = run_c(exe, windows, features)
    mismatches = compare(results, run_python(windows))
    return windows, sources, features, results, events, mismatches


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", nargs="+", help="ses dosyaları (wav/mp3/...), sırayla birleştirilir")
    p.add_argument("--gap", type=float, default=1.0, help="dosyalar arası sessizlik (sn)")
    p.add_argument("--dashboard", action="store_true", help="olayları panelin dosyalarına yaz")
    p.add_argument("--delay", type=float, default=0.0, help="--dashboard'da pencere başına bekleme (sn)")
    p.add_argument("--export-scenario", nargs="?", const=SCENARIO_PATH, default=None,
                   help=f"Wokwi senaryosu yaz (varsayılan {os.path.relpath(SCENARIO_PATH, ROOT)})")
    p.add_argument("--report", default=None, help="özet JSON dosyası")
    p.add_argument("--serial-out", default=None, help="seri çıktı biçiminde satırları bu dosyaya yaz")
    args = p.parse_args()

    windows, sources, features, results, events, mismatches = simulate(args.audio, args.gap)
    print_table(results, events, sources)

    alarms = [e for evs in events for e in evs if e["type"] == "alarm"]
    print(f"\n{len(results)} pencere, {len(alarms)} alarm. "
          f"C (cihaz kodu) ile Python aynı sonuç: {len(results) - len(mismatches)}/{len(results)}")
    for m in mismatches:
        print(f"  FARK pencere {m['window']}: C {m['c']} / Python {m['python']}")

    if args.report:
        per_file = {}
        for r, src in zip(results, sources):
            d = per_file.setdefault(src, {"windows": 0, "status": {}, "states": {}})
            d["windows"] += 1
            d["status"][r["status"]] = d["status"].get(r["status"], 0) + 1
            if r["state"] != "-":
                d["states"][r["state"]] = d["states"].get(r["state"], 0) + 1
        report = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "audio": [os.path.relpath(a, ROOT).replace(os.sep, "/") for a in args.audio],
                  "models": {k: os.path.relpath(v, ROOT).replace(os.sep, "/") for k, v in MODELS.items()},
                  "features": "python (librosa); C feature extraction not implemented (ESP-4)",
                  "windows": len(results), "c_python_agreement": len(results) - len(mismatches),
                  "mismatches": mismatches, "per_file": per_file,
                  "events": [e for evs in events for e in evs]}
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"-> {args.report}")
    if args.serial_out:
        with open(args.serial_out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(serial_lines(results, events)) + "\n")
        print(f"-> {args.serial_out}")
    if args.export_scenario:
        desc = " ".join(os.path.relpath(a, ROOT).replace(os.sep, "/") for a in args.audio)
        export_scenario(args.export_scenario, results, features, sources, desc)
    if args.dashboard:
        to_dashboard(results, events, args.delay)
        print("Olaylar panele yazıldı: streamlit run app.py")
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
