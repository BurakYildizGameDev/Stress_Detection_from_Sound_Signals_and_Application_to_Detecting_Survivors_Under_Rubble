import os
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.export_c_model import (
    c_float,
    export_random_forest_to_c,
    float32_threshold,
)


def _find_c_compiler():
    """CC ortam değişkeni (ör. "python -m ziglang cc") ya da PATH'teki gcc/cc/clang."""
    if os.environ.get("CC"):
        return shlex.split(os.environ["CC"], posix=os.name != "nt")
    for name in ("gcc", "cc", "clang"):
        path = shutil.which(name)
        if path:
            return [path]
    return None


CC = _find_c_compiler()
needs_cc = pytest.mark.skipif(CC is None, reason="C derleyicisi yok (CC ortam değişkeni ya da gcc/clang)")


def _fit(n_classes, n_features, n_estimators=5, max_depth=4, seed=42, scale=1.0):
    rng = np.random.default_rng(seed)
    X = (rng.standard_normal((200, n_features)) * scale).astype(np.float32)
    y = rng.integers(0, n_classes, size=200)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    clf.fit(X, y)
    return clf, X


def test_header_structure():
    clf, _ = _fit(3, 5)
    code = export_random_forest_to_c(clf, model_name="rubble_test", threshold=0.45,
                                     feature_spec={"sr": 16000, "version": 3, "desc": "test"})
    assert "#ifndef RUBBLE_TEST_MODEL_H" in code
    assert "#define RUBBLE_TEST_NUM_FEATURES 5" in code
    assert "#define RUBBLE_TEST_NUM_CLASSES 3" in code
    assert "#define RUBBLE_TEST_NUM_TREES 5" in code
    assert "#define RUBBLE_TEST_CLASS_0 0" in code
    assert "#define RUBBLE_TEST_THRESHOLD 0.45f" in code
    assert "#define RUBBLE_TEST_SAMPLE_RATE 16000" in code
    assert "static void rubble_test_tree_4(" in code
    assert "static int rubble_test_predict(" in code


def test_class_names_must_match_model_order():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3)).astype(np.float32)
    y = np.array(["stress", "normal", "panic"] * 20)
    clf = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=0).fit(X, y)

    code = export_random_forest_to_c(clf, model_name="m")  # classes_ alfabetik
    assert 'm_class_names[3] = {"normal", "panic", "stress"}' in code
    assert "#define M_CLASS_PANIC 1" in code
    with pytest.raises(ValueError):
        export_random_forest_to_c(clf, ["stress", "normal", "panic"], model_name="m")


def test_invalid_model_name_rejected():
    clf, _ = _fit(2, 3, n_estimators=1)
    with pytest.raises(ValueError):
        export_random_forest_to_c(clf, model_name="human-detector")


def test_c_float_literals_are_valid_c():
    assert c_float(3) == "3.0f"
    assert c_float(0.5) == "0.5f"
    assert c_float(1e-5) == "1e-05f"
    with pytest.raises(ValueError):
        c_float(float("inf"))


def test_float32_threshold_preserves_sklearn_decisions():
    # sklearn: float32 x ile float64 t karşılaştırılır. Aşağı yuvarlanan float32
    # eşik her float32 x için aynı yönü vermeli; en yakına yuvarlama vermez.
    rng = np.random.default_rng(1)
    a = rng.standard_normal(20000).astype(np.float32)
    b = np.nextafter(a, np.float32(np.inf))
    t = (a.astype(np.float64) + b.astype(np.float64)) / 2.0  # sklearn tarzı orta nokta
    t32 = np.array([float32_threshold(v) for v in t], dtype=np.float32)
    for x in (a, b):
        np.testing.assert_array_equal(x <= t, x <= t32)
    nearest = t.astype(np.float32)
    assert np.any((b <= t) != (b <= nearest))  # yuvarlama sorununun gerçek olduğunu göster


def _compile_and_run(tmp_path, headers, main_src, stdin_text):
    for name, code in headers.items():
        (tmp_path / name).write_text(code, encoding="utf-8")
    (tmp_path / "main.c").write_text(main_src, encoding="utf-8")
    exe = tmp_path / ("prog.exe" if os.name == "nt" else "prog")
    subprocess.run(CC + ["-std=c99", "-O1", "-Wall", "-Werror", "-Wno-unused-function",
                         "-o", str(exe), str(tmp_path / "main.c")],
                   check=True, capture_output=True, text=True)
    out = subprocess.run([str(exe)], input=stdin_text, check=True,
                         capture_output=True, text=True)
    return out.stdout


PREDICT_MAIN = r"""
#include <stdio.h>
#include "{header}"
int main(void) {{
    float f[{P}_NUM_FEATURES], p[{P}_NUM_CLASSES];
    for (;;) {{
        for (int i = 0; i < {P}_NUM_FEATURES; i++)
            if (scanf("%f", &f[i]) != 1) return 0;
        int best = {name}_predict(f, p);
        printf("%d", best);
        for (int c = 0; c < {P}_NUM_CLASSES; c++) printf(" %.9g", p[c]);
        printf("\n");
    }}
}}
"""


@needs_cc
@pytest.mark.parametrize("n_classes,scale", [(2, 1.0), (5, 1.0), (3, 1e-4)])
def test_compiled_c_matches_sklearn(tmp_path, n_classes, scale):
    # scale=1e-4: küçük ölçekli öznitelikler, %.6f ile yazılan eşiklerin bozulduğu durum
    clf, X = _fit(n_classes, 6, n_estimators=20, max_depth=8, scale=scale)
    X_test = np.vstack([X, (np.random.default_rng(7).standard_normal((200, 6)) * scale)]).astype(np.float32)

    code = export_random_forest_to_c(clf, model_name="m")
    stdin = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in X_test) + "\n"
    out = _compile_and_run(tmp_path, {"m_model.h": code},
                           PREDICT_MAIN.format(header="m_model.h", P="M", name="m"), stdin)

    rows = np.array([[float(v) for v in line.split()] for line in out.strip().splitlines()])
    c_best, c_probs = rows[:, 0].astype(int), rows[:, 1:]
    np.testing.assert_allclose(c_probs, clf.predict_proba(X_test), atol=1e-5)
    np.testing.assert_array_equal(c_best, np.argmax(c_probs, axis=1))


@needs_cc
def test_two_models_in_one_translation_unit(tmp_path):
    human, _ = _fit(2, 4, n_estimators=3)
    emerg, _ = _fit(5, 6, n_estimators=3)
    headers = {
        "human_model.h": export_random_forest_to_c(human, model_name="human"),
        "emerg_model.h": export_random_forest_to_c(emerg, model_name="emerg"),
    }
    main_src = r"""
#include <stdio.h>
#include "human_model.h"
#include "emerg_model.h"
int main(void) {
    float fh[HUMAN_NUM_FEATURES] = {0}, fe[EMERG_NUM_FEATURES] = {0};
    printf("%d %d %d %d\n", human_predict(fh, 0), emerg_predict(fe, 0),
           HUMAN_NUM_CLASSES, EMERG_NUM_CLASSES);
    return 0;
}
"""
    out = _compile_and_run(tmp_path, headers, main_src, "")
    h, e, nh, ne = map(int, out.split())
    assert (nh, ne) == (2, 5)
    zeros_h, zeros_e = np.zeros((1, 4), np.float32), np.zeros((1, 6), np.float32)
    assert h == int(np.argmax(human.predict_proba(zeros_h)))
    assert e == int(np.argmax(emerg.predict_proba(zeros_e)))
