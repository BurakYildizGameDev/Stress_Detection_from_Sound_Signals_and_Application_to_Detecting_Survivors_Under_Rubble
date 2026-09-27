import os
import sys

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.train_esp import alloc_section_sizes, measure_flash_bytes, select_best
from tests.test_c_model_export import CC, needs_cc


def _row(score, flash):
    return {"score": score, "flash_bytes": flash}


def test_select_best_respects_budget():
    rows = [_row(0.90, 900_000), _row(0.85, 400_000), _row(0.80, 100_000)]
    assert select_best(rows, 500_000) is rows[1]
    assert select_best(rows, 1_000_000) is rows[0]
    assert select_best(rows, 50_000) is None


def test_select_best_prefers_smaller_on_tie():
    rows = [_row(0.8500001, 400_000), _row(0.85, 200_000)]
    assert select_best(rows, 500_000) is rows[1]


def test_alloc_section_sizes_rejects_non_elf():
    with pytest.raises(ValueError):
        alloc_section_sizes(b"MZ\x90\x00not an elf")


@needs_cc
def test_measured_flash_grows_with_model_size():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 5)).astype(np.float32)
    y = rng.integers(0, 2, size=400)
    small = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=0).fit(X, y)
    large = RandomForestClassifier(n_estimators=10, max_depth=8, random_state=0).fit(X, y)
    try:
        s, l = measure_flash_bytes(small, CC), measure_flash_bytes(large, CC)
    except ValueError:
        pytest.skip("derleyici ELF üretmiyor (ör. Windows COFF)")
    assert 0 < s < l
