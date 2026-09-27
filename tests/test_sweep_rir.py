import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sweep_rir import generate_log_sweep_and_inverse, deconvolve_rubble_response


def test_sweep_generation_shape_and_finite():
    sr = 16000
    duration = 2.0
    sweep, inv_filter = generate_log_sweep_and_inverse(
        f1=50.0, f2=4000.0, duration_sec=duration, sr=sr
    )
    expected_len = int(sr * duration)
    assert len(sweep) == expected_len
    assert len(inv_filter) == expected_len
    assert np.all(np.isfinite(sweep))
    assert np.all(np.isfinite(inv_filter))


def test_sweep_deconvolution_impulse_recovery():
    sr = 16000
    sweep, inv_filter = generate_log_sweep_and_inverse(
        f1=50.0, f2=4000.0, duration_sec=2.0, sr=sr
    )

    # İdeal sistemde (gecikmesiz doğrudan yol) sweep'i doğrudan dekonvolüe edelim
    recovered_rir = deconvolve_rubble_response(sweep, inv_filter, sr=sr, max_rir_sec=0.2)

    assert len(recovered_rir) > 0
    assert np.all(np.isfinite(recovered_rir))
    # Tepe noktası 1.0 normalize olmalı
    assert np.max(np.abs(recovered_rir)) == pytest.approx(1.0, abs=1e-4)

    # Tepe noktası civarındaki enerji, toplam enerjinin büyük bölümünü oluşturmalı (Dirac delta)
    peak_idx = int(np.argmax(np.abs(recovered_rir)))
    peak_energy = recovered_rir[peak_idx] ** 2
    margin_energy = np.sum(recovered_rir[max(0, peak_idx - 10):peak_idx + 10] ** 2)
    assert margin_energy > 0.5 * peak_energy


def test_deconvolve_with_delayed_response():
    sr = 16000
    sweep, inv_filter = generate_log_sweep_and_inverse(
        f1=100.0, f2=2000.0, duration_sec=1.5, sr=sr
    )

    # 50 ms gecikmeli ve 0.5 genlikli yapay enkaz sinyali
    delay_samples = int(0.050 * sr)
    delayed_signal = np.pad(sweep * 0.5, (delay_samples, 0), mode="constant")

    rir = deconvolve_rubble_response(delayed_signal, inv_filter, sr=sr, max_rir_sec=0.25)
    assert len(rir) == int(sr * 0.25)
    assert np.all(np.isfinite(rir))
    assert np.max(np.abs(rir)) == pytest.approx(1.0, abs=1e-4)


def test_sweep_empty_or_silence():
    sr = 16000
    _, inv = generate_log_sweep_and_inverse(duration_sec=0.5, sr=sr)
    rir_empty = deconvolve_rubble_response(np.array([], dtype=np.float32), inv, sr=sr)
    assert len(rir_empty) == int(sr * 0.5)

    rir_zero = deconvolve_rubble_response(np.zeros(sr, dtype=np.float32), inv, sr=sr)
    assert np.all(np.isfinite(rir_zero))
