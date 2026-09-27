import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rubble_acoustics import (
    viscoelastic_damping,
    apply_cavity_resonance,
    generate_rubble_rir,
    generate_rubble_ambient_noise,
    apply_rubble_acoustics
)
from augment_rubble import RUBBLE_CONDITIONS, augment_with_rubble


def tone(sr, sec=1.0, freq=440.0):
    t = np.arange(int(sr * sec)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_viscoelastic_damping_frequency_dependency():
    sr = 16000
    # 150 Hz alçak frekans (inilti/boğaz)
    y_low = tone(sr, sec=1.0, freq=150.0)
    # 3000 Hz yüksek frekans (tiz çığlık)
    y_high = tone(sr, sec=1.0, freq=3000.0)

    damped_low = viscoelastic_damping(y_low, sr, distance_m=3.0)
    damped_high = viscoelastic_damping(y_high, sr, distance_m=3.0)

    p_low_orig = np.mean(y_low ** 2)
    p_low_damped = np.mean(damped_low ** 2)
    ratio_low = p_low_damped / p_low_orig

    p_high_orig = np.mean(y_high ** 2)
    p_high_damped = np.mean(damped_high ** 2)
    ratio_high = p_high_damped / p_high_orig

    # Düşük frekans yüksek frekanstan ÇOK DAHA AZ sönümlenmelidir
    assert ratio_low > 0.50
    assert ratio_high < 0.01
    assert ratio_low > 50.0 * ratio_high


def test_cavity_resonance_boosts_modes():
    sr = 16000
    # 175 Hz rezonans moduna denk gelen ton
    y_mode = tone(sr, sec=1.0, freq=175.0)
    # 1500 Hz rezonans dışı ton
    y_off = tone(sr, sec=1.0, freq=1500.0)

    res_mode = apply_cavity_resonance(y_mode, sr)
    res_off = apply_cavity_resonance(y_off, sr)

    gain_mode = np.sqrt(np.mean(res_mode ** 2)) / np.sqrt(np.mean(y_mode ** 2))
    gain_off = np.sqrt(np.mean(res_off ** 2)) / np.sqrt(np.mean(y_off ** 2))

    assert gain_mode > gain_off
    assert gain_mode > 1.2


def test_rubble_rir_properties():
    sr = 16000
    rir = generate_rubble_rir(sr, distance_m=2.0, duration_sec=0.15)
    assert len(rir) == int(sr * 0.15)
    assert np.all(np.isfinite(rir))
    assert np.max(np.abs(rir)) == pytest.approx(1.0, abs=1e-4)


def test_ambient_noise_snr():
    sr = 16000
    sig = tone(sr, sec=1.0, freq=300.0)
    target_snr = 15.0
    noise = generate_rubble_ambient_noise(sr, duration_sec=1.0, snr_target_db=target_snr, ref_signal=sig)

    p_sig = np.mean(sig ** 2)
    p_noise = np.mean(noise ** 2)
    measured_snr = 10.0 * np.log10(p_sig / p_noise)

    assert measured_snr == pytest.approx(target_snr, abs=1.5)


def test_full_pipeline_shape_and_finite():
    sr = 16000
    y = tone(sr, sec=1.0, freq=400.0)
    out = apply_rubble_acoustics(y, sr, distance_m=2.0, noise_snr_db=12.0)

    assert out.shape == y.shape
    assert np.all(np.isfinite(out))


def test_rubble_conditions_augmentations():
    sr = 16000
    y = tone(sr, sec=1.0, freq=250.0)
    rng = np.random.default_rng(42)

    for cond_name, fn in RUBBLE_CONDITIONS.items():
        res = fn(y, sr, rng)
        assert res.shape == y.shape
        assert np.all(np.isfinite(res))

    res_preset = augment_with_rubble(y, sr, condition="rubble_medium", rng=rng)
    assert res_preset.shape == y.shape
    assert np.all(np.isfinite(res_preset))


def test_edge_cases():
    y_empty = np.array([], dtype=np.float32)
    assert len(apply_rubble_acoustics(y_empty, 16000)) == 0
    assert len(viscoelastic_damping(y_empty, 16000)) == 0

    y_zero = np.zeros(16000, dtype=np.float32)
    out_zero = apply_rubble_acoustics(y_zero, 16000, noise_snr_db=None)
    assert np.all(np.isfinite(out_zero))
