"""
augment_rubble.py - Gerçekçi Enkaz Akustiği Veri Artırma Modülü.

Eğitim ve değerlendirme aşamalarında modellerin moloz arkasından gelen boğuk,
yankılanmış ve gürültülü seslere dayanıklılığını artırmak için kullanılır.
"""
import numpy as np
from rubble_acoustics import apply_rubble_acoustics

RUBBLE_CONDITIONS = {
    # Hafif enkaz: 1.0 metre moloz, rezonans var, temiz
    "rubble_mild": lambda y, sr, rng: apply_rubble_acoustics(
        y, sr, distance_m=1.0, void_resonance=True, rir_conv=True, noise_snr_db=None, rng=rng
    ),
    # Orta enkaz: 2.5 metre moloz, rezonans + 15 dB saha gürültüsü
    "rubble_medium": lambda y, sr, rng: apply_rubble_acoustics(
        y, sr, distance_m=2.5, void_resonance=True, rir_conv=True, noise_snr_db=15.0, rng=rng
    ),
    # Ağır enkaz: 4.0 metre derin moloz, aşırı tiz kaybı, 8 dB jeneratör uğultusu
    "rubble_severe": lambda y, sr, rng: apply_rubble_acoustics(
        y, sr, distance_m=4.0, void_resonance=True, rir_conv=True, noise_snr_db=8.0, rng=rng
    ),
}


def augment_with_rubble(y, sr, condition="rubble_medium", rng=None):
    """
    Belirtilen enkaz profiline göre sesi fiziksel moloz filtresinden geçirir.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    fn = RUBBLE_CONDITIONS.get(condition, RUBBLE_CONDITIONS["rubble_medium"])
    return fn(y, sr, rng)
