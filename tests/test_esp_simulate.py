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
