import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import (EMERGENCY_N_FEATURES, EMERGENCY_SR, HUMAN_N_FEATURES,
                      HUMAN_SR, emergency_features, human_features)


def tone(sr, sec=1.0, freq=440.0):
    t = np.arange(int(sr * sec)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_human_feature_shape():
    f = human_features(tone(HUMAN_SR), HUMAN_SR)
    assert f.shape == (HUMAN_N_FEATURES,)
    assert np.all(np.isfinite(f))


def test_emergency_feature_shape():
    f = emergency_features(tone(EMERGENCY_SR), EMERGENCY_SR)
    assert f.shape == (EMERGENCY_N_FEATURES,)
    assert np.all(np.isfinite(f))


def test_emergency_features_resample_input():
    # Canlı pipeline 22.05 kHz verir; model 16 kHz görmüştür.
    from_22k = emergency_features(tone(HUMAN_SR), HUMAN_SR)
    native = emergency_features(tone(EMERGENCY_SR), EMERGENCY_SR)
    # Spectral centroid 440 Hz tonda örnekleme hızından bağımsız olmalı
    assert from_22k[-1] == pytest.approx(native[-1], rel=0.05)


def test_saved_emergency_features_width():
    X_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "features", "X_emergency.npy")
    if not os.path.exists(X_path):
        pytest.skip("features/X_emergency.npy yok")
    assert np.load(X_path).shape[1] == EMERGENCY_N_FEATURES
