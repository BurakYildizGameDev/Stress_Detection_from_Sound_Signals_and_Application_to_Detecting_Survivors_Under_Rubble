import pytest
from dataset import _ravdess, _berlin, Skip

def test_ravdess_parsing():
    spk, emo, cid = _ravdess('03-01-05-01-01-01-01.wav', None)
    assert spk == 'actor01'
    assert emo == 'angry'

def test_berlin_parsing():
    spk, emo, cid = _berlin('03a01Fa.wav', None)
    assert spk == '03'
    assert emo == 'happy'
