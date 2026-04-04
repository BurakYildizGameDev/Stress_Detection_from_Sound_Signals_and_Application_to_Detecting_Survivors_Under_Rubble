import pytest
from pipeline import load_models
def test_load_models():
    models = load_models()
    assert 'human' in models and 'emergency' in models

from pipeline import analyze_window
import numpy as np
def test_silence():
    res = analyze_window(np.zeros(22050, dtype=np.float32))
    assert res['status'] == 'silence'

def test_no_human():
    pass

def test_detected():
    pass

def test_regression():
    pass
