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

from dataset import _esc50
def test_esc50_vocal_exclusion():
    with pytest.raises(Skip):
        _esc50('1-10021-A-20.wav', None)

from dataset import trim_silence
def test_trim_silence_all_silent():
    y = np.zeros(16000, dtype=np.float32)
    trimmed = trim_silence(y)
    assert len(trimmed) == 16000
