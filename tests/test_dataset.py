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

from augment import add_noise_snr
def test_add_noise_snr():
    rng = np.random.default_rng(0)
    y = np.ones(1000, dtype=np.float32)
    noisy = add_noise_snr(y, 20, rng)
    assert len(noisy) == len(y)

from augment import lowpass
def test_lowpass():
    y = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    filtered = lowpass(y, 16000, 400)
    assert not np.isnan(filtered).any()

from augment import gain_db
def test_gain():
    y = np.ones(100, dtype=np.float32)
    scaled = gain_db(y, -6)
    assert np.isclose(scaled[0], 0.501187, atol=1e-3)
