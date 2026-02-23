import pytest
from pipeline import load_models
def test_load_models():
    models = load_models()
    assert 'human' in models and 'emergency' in models
