import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from whisper_converter import convert_to_whisper_dsp, convert_to_moan_dsp
from features_v2 import extract_hnr_robust, emergency_features_v2, human_features_v2


def speech_like_signal(sr=16000, sec=1.0):
    t = np.arange(int(sr * sec)) / sr
    # Formant benzeri 3 harmonik
    sig = (
        0.5 * np.sin(2 * np.pi * 180 * t) +
        0.3 * np.sin(2 * np.pi * 540 * t) +
        0.2 * np.sin(2 * np.pi * 1200 * t)
    )
    # Zarf
    env = 0.5 * (1 - np.cos(2 * np.pi * t / sec))
    return (sig * env).astype(np.float32)


def test_convert_to_whisper_dsp_output_properties():
    sr = 16000
    y = speech_like_signal(sr=sr, sec=1.0)
    w = convert_to_whisper_dsp(y, sr=sr)

    assert w.shape == y.shape
    assert np.all(np.isfinite(w))
    # Fısıltı enerjisi orijinalden düşük olmalı
    rms_orig = np.sqrt(np.mean(y ** 2))
    rms_w = np.sqrt(np.mean(w ** 2))
    assert rms_w < rms_orig

    # Fısıltıda HNR sıfıra yakın veya çok düşük olmalı
    hnr_w = extract_hnr_robust(w, sr)
    hnr_orig = extract_hnr_robust(y, sr)
    assert hnr_w < hnr_orig


def test_convert_to_moan_dsp_output_properties():
    sr = 16000
    y = speech_like_signal(sr=sr, sec=1.0)
    m = convert_to_moan_dsp(y, sr=sr)

    assert m.shape == y.shape
    assert np.all(np.isfinite(m))

    # İnlemede 0-500 Hz enerji oranı çok yüksek olmalıdır
    f_orig = emergency_features_v2(y, sr)
    f_moan = emergency_features_v2(m, sr)
    # Index 16: low_ratio
    assert f_moan[16] > f_orig[16]
    assert f_moan[16] > 0.70


def test_converters_empty_or_silence():
    y_empty = np.array([], dtype=np.float32)
    assert len(convert_to_whisper_dsp(y_empty)) == 0
    assert len(convert_to_moan_dsp(y_empty)) == 0

    y_zero = np.zeros(16000, dtype=np.float32)
    w_zero = convert_to_whisper_dsp(y_zero)
    m_zero = convert_to_moan_dsp(y_zero)
    assert np.all(np.isfinite(w_zero))
    assert np.all(np.isfinite(m_zero))


def test_converters_are_deterministic_with_rng():
    sr = 16000
    y = speech_like_signal(sr=sr, sec=1.0)
    a = convert_to_whisper_dsp(y, sr=sr, rng=np.random.default_rng(7))
    b = convert_to_whisper_dsp(y, sr=sr, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
