import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features_v2 import (
    HUMAN_N_FEATURES_V2,
    EMERGENCY_N_FEATURES_V2,
    HUMAN_SR,
    EMERGENCY_SR,
    extract_hnr_robust,
    human_features_v2,
    emergency_features_v2,
    FEATURE_SPECS_V2
)


def tone(sr, sec=1.0, freq=440.0):
    t = np.arange(int(sr * sec)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def white_noise(sr, sec=1.0):
    rng = np.random.default_rng(42)
    return (0.1 * rng.standard_normal(int(sr * sec))).astype(np.float32)


def test_human_v2_shape_and_finite():
    y = tone(HUMAN_SR, sec=1.0, freq=220.0)
    feats = human_features_v2(y, HUMAN_SR)
    assert feats.shape == (HUMAN_N_FEATURES_V2,)
    assert np.all(np.isfinite(feats))


def test_emergency_v2_shape_and_finite():
    y = tone(EMERGENCY_SR, sec=1.0, freq=300.0)
    feats = emergency_features_v2(y, EMERGENCY_SR)
    assert feats.shape == (EMERGENCY_N_FEATURES_V2,)
    assert np.all(np.isfinite(feats))


def test_hnr_distinguishes_periodic_from_noise():
    sr = 16000
    periodic = tone(sr, sec=1.0, freq=150.0)
    noise = white_noise(sr, sec=1.0)

    hnr_periodic = extract_hnr_robust(periodic, sr)
    hnr_noise = extract_hnr_robust(noise, sr)

    assert hnr_periodic > 5.0
    assert hnr_noise < 2.0


def test_spectral_flatness_detects_whisper_noise():
    # Beyaz gürültü (fısıltı modeli) yüksek spectral flatness verir
    y_noise = white_noise(HUMAN_SR, sec=1.0)
    feats_noise = human_features_v2(y_noise, HUMAN_SR)

    # 440 Hz saf ton düşük spectral flatness verir
    y_tone = tone(HUMAN_SR, sec=1.0, freq=440.0)
    feats_tone = human_features_v2(y_tone, HUMAN_SR)

    # Index 27: flatness_mean
    assert feats_noise[27] > feats_tone[27]
    assert feats_noise[27] > 0.3


def test_sub_band_ratio_detects_low_frequency_moan():
    # 120 Hz derin inleme simülasyonu
    y_moan = tone(EMERGENCY_SR, sec=1.0, freq=120.0)
    feats_moan = emergency_features_v2(y_moan, EMERGENCY_SR)

    # 2500 Hz tiz ıslık / çığlık
    y_high = tone(EMERGENCY_SR, sec=1.0, freq=2500.0)
    feats_high = emergency_features_v2(y_high, EMERGENCY_SR)

    # Index 16: low_ratio (0-500 Hz oranı)
    assert feats_moan[16] > 0.85
    assert feats_high[16] < 0.10


def test_resample_cross_sr():
    # 22.05 kHz girdi ile emergency_features_v2 çağrısı
    y_22k = tone(HUMAN_SR, sec=1.0, freq=300.0)
    f1 = emergency_features_v2(y_22k, sr=HUMAN_SR)
    assert f1.shape == (EMERGENCY_N_FEATURES_V2,)
    assert np.all(np.isfinite(f1))

    # 16 kHz girdi ile human_features_v2 çağrısı
    y_16k = tone(EMERGENCY_SR, sec=1.0, freq=300.0)
    f2 = human_features_v2(y_16k, sr=EMERGENCY_SR)
    assert f2.shape == (HUMAN_N_FEATURES_V2,)
    assert np.all(np.isfinite(f2))


def test_silence_all_zeros():
    # Tam sıfır sinyal geldiğinde NaN veya Inf patlamamalı
    y_zero = np.zeros(HUMAN_SR, dtype=np.float32)
    f_human = human_features_v2(y_zero, HUMAN_SR)
    f_emerg = emergency_features_v2(y_zero, HUMAN_SR)

    assert f_human.shape == (HUMAN_N_FEATURES_V2,)
    assert f_emerg.shape == (EMERGENCY_N_FEATURES_V2,)
    assert np.all(np.isfinite(f_human))
    assert np.all(np.isfinite(f_emerg))


def test_features_are_loudness_invariant():
    # Aynı sinyal 40 dB daha sessiz geldiğinde öznitelikler değişmemeli; aksi
    # hâlde model "sessizse fısıltıdır" kısayolunu öğrenir.
    y = tone(EMERGENCY_SR, sec=1.0, freq=200.0) + white_noise(EMERGENCY_SR, sec=1.0)
    quiet = (y * 0.01).astype(np.float32)
    np.testing.assert_allclose(emergency_features_v2(y), emergency_features_v2(quiet),
                               rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(human_features_v2(y, EMERGENCY_SR),
                               human_features_v2(quiet, EMERGENCY_SR), rtol=1e-3, atol=1e-3)
