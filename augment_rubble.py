"""
augment_rubble.py - Gerçekçi Enkaz Akustiği Veri Artırma Modülü.

Eğitim ve değerlendirme aşamalarında modellerin moloz arkasından gelen boğuk,
yankılanmış ve gürültülü seslere dayanıklılığını artırmak için kullanılır.
"""
import numpy as np
from rubble_acoustics import apply_rubble_acoustics, random_cavity_modes


def random_rubble(y, sr, rng):
    """Eğitim artırması: mesafe, rezonans modları ve gürültü her çağrıda
    rastgeledir. Değerlendirmedeki sabit mild/medium/severe koşullarıyla aynı
    parametreler hiçbir zaman ezberlenmez."""
    snr = None if rng.random() < 0.2 else float(rng.uniform(5.0, 25.0))
    return apply_rubble_acoustics(
        y, sr, distance_m=float(rng.uniform(0.3, 4.0)),
        void_resonance=bool(rng.random() < 0.8), rir_conv=True,
        noise_snr_db=snr, rng=rng, modes=random_cavity_modes(rng))

RUBBLE_CONDITIONS = {
    "rubble_random": random_rubble,
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
