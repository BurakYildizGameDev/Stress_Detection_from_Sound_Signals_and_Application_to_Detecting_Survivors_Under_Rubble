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



def test_augmentations_and_conditions_keep_audio_valid():
    from augment import AUGMENTATIONS, CONDITIONS
    y = tone(HUMAN_SR)
    fns = [*AUGMENTATIONS["human"].values(), *AUGMENTATIONS["non_human"].values(), *CONDITIONS.values()]
    for fn in fns:
        out = fn(y, HUMAN_SR, np.random.default_rng(0))
        assert out.shape == y.shape and np.all(np.isfinite(out))


def test_lowpass_attenuates_high_frequencies():
    from augment import lowpass
    high = tone(HUMAN_SR, freq=4000)
    assert np.sqrt(np.mean(lowpass(high, HUMAN_SR, 400) ** 2)) < 0.01
