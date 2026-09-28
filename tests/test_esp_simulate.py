"""
Bilgisayar simülatörü (scripts/esp_simulate.py) ve Wokwi senaryosu
(firmware/include/sim_scenario.h) testleri. C derleyicisi gerekir.
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.esp_simulate import CORE_DIR, INCLUDE_DIR, ROOT, simulate
from tests.test_c_model_export import CC, needs_cc


@needs_cc
def test_simulator_matches_python_pipeline_on_real_audio():
    # Cihazın C kodu (RMS, modeller, karar) ile pipeline_v2 aynı ESP modelleriyle
    # her pencerede aynı sonucu vermeli.
    paths = [os.path.join(ROOT, "samples", f) for f in ("people_talk.mp3", "bird.mp3", "car_start.mp3")]
    windows, sources, features, results, events, mismatches = simulate(paths, gap_sec=1.0, cc=CC)
    assert mismatches == []
    assert len(results) == len(windows) >= 15
    statuses = {r["status"] for r in results}
    assert {"silence", "no_human", "DETECTED"} <= statuses  # araya konan sessizlik de dahil


MODEL_CHECK = r"""
#include <stdio.h>
#include "human_detector_model.h"
#include "emergency_classifier_model.h"
int main(void) {
    char kind[4];
    float f[64], p[8];
    while (scanf("%3s", kind) == 1) {
        int emerg = kind[0] == 'E';
        int nf = emerg ? EMERGENCY_CLASSIFIER_NUM_FEATURES : HUMAN_DETECTOR_NUM_FEATURES;
        for (int i = 0; i < nf; i++) if (scanf("%f", &f[i]) != 1) return 2;
        int nc = emerg ? EMERGENCY_CLASSIFIER_NUM_CLASSES : HUMAN_DETECTOR_NUM_CLASSES;
        int best = emerg ? emergency_classifier_predict(f, p) : human_detector_predict(f, p);
        printf("%d", best);
        for (int c = 0; c < nc; c++) printf(" %.9g", p[c]);
        printf("\n");
    }
    return 0;
}
"""


@needs_cc
def test_committed_esp_headers_match_sklearn_on_real_features(tmp_path):
    # Repodaki model başlıkları, ESP .pkl'leriyle gerçek ses özniteliklerinden
    # türetilen girdilerde aynı olasılığı ve aynı sınıfı vermeli (yakın olasılıklı
    # beraberlik adayları dahil).
    import joblib
    import numpy as np
    from scripts.esp_simulate import MODELS, load_windows, window_features

    paths = [os.path.join(ROOT, "samples", f) for f in
             ("woman_scream.mp3", "people_talk.mp3", "bird.mp3", "car_start.mp3")]
    windows, _ = load_windows(paths, gap_sec=0)
    feats = [window_features(y) for y in windows]
    rng = np.random.default_rng(0)

    def perturb(base, n):
        rows = base[rng.integers(0, len(base), n)]
        return (rows * (1 + rng.normal(0, 0.05, rows.shape))).astype(np.float32)

    H = perturb(np.array([h for h, _ in feats]), 1000)
    E = perturb(np.array([e for _, e in feats]), 1000)

    src = tmp_path / "models.c"
    src.write_text(MODEL_CHECK, encoding="utf-8")
    exe = tmp_path / ("models.exe" if os.name == "nt" else "models")
    subprocess.run(CC + ["-std=c99", "-O1", "-I", INCLUDE_DIR, "-o", str(exe), str(src)],
                   check=True, capture_output=True, text=True)
    stdin = "".join("H " + " ".join(f"{v:.9g}" for v in r) + "\n" for r in H)
    stdin += "".join("E " + " ".join(f"{v:.9g}" for v in r) + "\n" for r in E)
    out = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, check=True).stdout
    rows = [line.split() for line in out.strip().splitlines()]
    c_best = np.array([int(r[0]) for r in rows])

    for X, key, sl in ((H, "human", slice(0, 1000)), (E, "emergency", slice(1000, 2000))):
        model = joblib.load(MODELS[key])
        c_probs = np.array([[float(v) for v in r[1:]] for r in rows[sl]])
        py_probs = model.predict_proba(X)
        np.testing.assert_allclose(c_probs, py_probs, atol=1e-5)
        np.testing.assert_array_equal(c_best[sl], np.argmax(py_probs, axis=1))


SCENARIO_CHECK = r"""
#include <stdio.h>
#include <string.h>
#include "rubble_pipeline.h"
#include "sim_scenario.h"

static int f_human(void* ctx, float* out) {
    memcpy(out, ((const sim_window_t*)ctx)->human, sizeof(((const sim_window_t*)ctx)->human));
    return RUBBLE_FEATURES_OK;
}
static int f_emergency(void* ctx, float* out) {
    memcpy(out, ((const sim_window_t*)ctx)->emergency, sizeof(((const sim_window_t*)ctx)->emergency));
    return RUBBLE_FEATURES_OK;
}

int main(void) {
    int mismatches = 0, detected = 0;
    for (int i = 0; i < SIM_SCENARIO_WINDOWS; i++) {
        const sim_window_t* w = &SIM_SCENARIO[i];
        rubble_result_t r;
        rubble_classify(w->rms, f_human, f_emergency, (void*)w, &r);
        if ((int)r.status != w->expected_status || r.state != w->expected_state) mismatches++;
        if (r.status == RUBBLE_DETECTED) detected++;
    }
    printf("%d %d %d\n", SIM_SCENARIO_WINDOWS, mismatches, detected);
    return 0;
}
"""


@needs_cc
def test_wokwi_scenario_is_consistent_with_current_models(tmp_path):
    # Wokwi'de cihaz her pencereyi expected_* ile karşılaştırır. Modeller ya da
    # karar kodu değişip senaryo yeniden üretilmezse bu test yakalar.
    src = tmp_path / "check.c"
    src.write_text(SCENARIO_CHECK, encoding="utf-8")
    exe = tmp_path / ("check.exe" if os.name == "nt" else "check")
    c_files = [os.path.join(CORE_DIR, f) for f in sorted(os.listdir(CORE_DIR)) if f.endswith(".c")]
    subprocess.run(CC + ["-std=c99", "-O1", "-I", INCLUDE_DIR, "-I", CORE_DIR, "-o", str(exe),
                         str(src)] + c_files + ["-lm"], check=True, capture_output=True, text=True)
    n, mismatches, detected = map(int, subprocess.run([str(exe)], capture_output=True, text=True,
                                                      check=True).stdout.split())
    assert mismatches == 0, "sim_scenario.h eski: esp_simulate.py --export-scenario ile yeniden üretin"
    assert n >= 10 and detected >= 3
