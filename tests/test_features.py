import numpy as np
import pytest
from features import human_features, emergency_features, HUMAN_SR, EMERGENCY_SR

def test_feature_shapes():
    y = np.zeros(HUMAN_SR, dtype=np.float32)
    h = human_features(y, HUMAN_SR)
    assert h.shape == (28,)
