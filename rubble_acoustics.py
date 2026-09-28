"""
rubble_acoustics.py - Gerçek Enkaz Akustiği, RIR Konvolüsyonu ve Fiziksel Moloz Sönümleme Motoru.

Bu modül, arama kurtarma sahasında katı betonarme molozların ve yaşam üçgeni hava ceplerinin
ses üzerindeki fiziksel etkilerini modeller:

Not: Bu modül ölçülmüş bir enkaz modeli değildir; fizikten esinlenen ve
katsayıları kalibre edilmemiş bir veri artırma (domain randomization) aracıdır.

1. Ampirik frekansa bağlı sönüm:
   12*f^1.5 + 6*f^2 dB/m (f kHz), 90 dB'de kırpılır. 1 m'de ~100 Hz: 0.4 dB,
   ~1 kHz: 18 dB, ~3 kHz: 116 dB (kırpılır). Gözenekli ortam için gerçek bir
   model (ör. Johnson-Champoux-Allard) ölçümle kalibre edilmelidir.
2. Boşluk rezonansı EQ'su:
   Gauss biçimli tepe filtreleri (varsayılan 115/175/225 Hz). Eğitimde
   random_cavity_modes() ile rastgeleleştirilir ki model sabit bir imzayı ezberlemesin.
3. Parametrik Enkaz İmpuls Yanıtı (Rubble Impulse Response - RIR):
   Beton yüzeylerden erken yansımalar ve moloz içi saçılma (diffuse scattering) ile
   gerçekçi yankı ve faz dağılması.
4. Saha Gürültü Katmanı:
   Jeneratör 50 Hz şebeke uğultusu ve harmonikleri, kompresör rölantisi ve moloz tozu sürtünmesi.
"""
import numpy as np
from scipy.signal import fftconvolve


def viscoelastic_damping(y, sr, distance_m=2.0):
    """
    Beton ve moloz içinde frekansın karesiyle orantılı viskoelastik zayıflama uygular.

    Parametreler:
        y (np.ndarray): Ses sinyali
        sr (int): Örnekleme hızı
        distance_m (float): Moloz içindeki mesafe (metre)
    Dönüş:
        np.ndarray: Yüksek frekansları sönümlenmiş ses sinyali
    """
    if len(y) == 0 or distance_m <= 0:
        return y.copy().astype(np.float32)

    n_samples = len(y)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)

    # Frekansa bağlı dB kaybı (Stokes-Kirchhoff modeli)
    # f = 100 Hz -> ~0.6 dB/m
    # f = 1000 Hz -> ~18 dB/m
    # f = 3000 Hz -> ~75 dB/m (yüksek frekanslar yok olur)
    f_khz = freqs / 1000.0
    attenuation_db_per_m = 12.0 * (f_khz ** 1.5) + 6.0 * (f_khz ** 2)
    total_attenuation_db = np.clip(attenuation_db_per_m * distance_m, 0.0, 90.0)

    # Lineer kazanç katsayısı
    h_gain = 10.0 ** (-total_attenuation_db / 20.0)

    # Frekans uzayında filtreleme
    y_fft = np.fft.rfft(y)
    y_damped = np.fft.irfft(y_fft * h_gain, n=n_samples)

    return y_damped.astype(np.float32)


def apply_cavity_resonance(y, sr, modes=((115.0, 8.0, 4.0), (175.0, 6.0, 3.5), (225.0, 4.0, 3.0))):
    """
    Enkaz boşluğundaki (yaşam üçgeni) duran dalga rezonans modlarını simüle eder.
    Her mod: (frekans_hz, kazanç_db, q_faktörü)
    """
    if len(y) == 0:
        return y.copy().astype(np.float32)

    n_samples = len(y)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)

    # Rezonans spektral zarfı
    gain_curve = np.ones_like(freqs, dtype=np.float64)

    for f_res, gain_db, q in modes:
        sigma = f_res / (2.0 * q)
        # Gauss biçimli rezonans tepesi
        peak = (10.0 ** (gain_db / 20.0) - 1.0) * np.exp(-((freqs - f_res) ** 2) / (2.0 * (sigma ** 2)))
        gain_curve += peak

    y_fft = np.fft.rfft(y)
    y_res = np.fft.irfft(y_fft * gain_curve, n=n_samples)

    # Enerji ölçekleme koruması
    rms_orig = np.sqrt(np.mean(y ** 2)) + 1e-12
    rms_res = np.sqrt(np.mean(y_res ** 2)) + 1e-12
    if rms_res > 2.0 * rms_orig:
        y_res = y_res * (1.5 * rms_orig / rms_res)

    return y_res.astype(np.float32)


def generate_rubble_rir(sr, distance_m=2.0, duration_sec=0.15, rng=None):
    """
    Betonarme moloz yığını ve hava cebi için parametrik İmpuls Yanıtı (RIR) üretir.
    - Doğrudan geliş (direct arrival)
    - Katı sınırlardan mikro-gecikmeli erken yansımalar (early reflections)
    - Moloz içi saçılma kuyruğu (diffuse scattering decay)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = int(sr * duration_sec)
    rir = np.zeros(n_samples, dtype=np.float32)

    # 1. Doğrudan varış (havadaki ses hızı ~343 m/s)
    direct_delay = max(1, int(distance_m / 343.0 * sr))
    if direct_delay < n_samples:
        rir[direct_delay] = 1.0

    # 2. Erken Yansımalar (Boşluktaki betonarme yansımalar: 2 ms - 20 ms)
    reflection_times_ms = [2.5, 5.0, 9.0, 14.5, 21.0]
    for ms in reflection_times_ms:
        delay = direct_delay + int(ms * 1e-3 * sr)
        if delay < n_samples:
            # Yansıma katsayısı ve mesafe kaybı
            amp = rng.uniform(0.15, 0.45) * (-1.0 if rng.random() > 0.5 else 1.0)
            rir[delay] += amp

    # 3. Moloz Saçılma Kuyruğu (Üstel sönümlü saçılma gürültüsü)
    t = np.arange(n_samples) / sr
    decay_time = 0.04 + 0.02 * distance_m  # Enkazda yankı çok kısadır (yüksek emilim)
    tail = rng.standard_normal(n_samples).astype(np.float32) * np.exp(-t / decay_time)
    tail[:direct_delay] = 0.0

    rir += tail * 0.25

    # Enerji normalizasyonu
    rir_norm = rir / (np.max(np.abs(rir)) + 1e-8)
    return rir_norm.astype(np.float32)


def generate_rubble_ambient_noise(sr, duration_sec, snr_target_db=15.0, ref_signal=None, rng=None):
    """
    Gerçek arama-kurtarma sahası gürültü profili:
    - 50 Hz dizel jeneratör uğultusu + 100/150/200 Hz harmonikleri
    - Ağır kurtarma aracı / hidrolik kompresör düşük frekans titreşimi (<80 Hz)
    - Zayıf moloz sürtünme hışırtısı
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = int(sr * duration_sec)
    t = np.arange(n_samples) / sr

    # 1. 50 Hz Jeneratör Şebeke Hum ve Harmonikleri
    hum = (
        0.50 * np.sin(2 * np.pi * 50.0 * t) +
        0.25 * np.sin(2 * np.pi * 100.0 * t) +
        0.15 * np.sin(2 * np.pi * 150.0 * t) +
        0.10 * np.sin(2 * np.pi * 200.0 * t)
    ).astype(np.float32)

    # 2. Kompresör / Hidrolik Titreşim (Pembe/Kahverengi gürültü)
    white = rng.standard_normal(n_samples).astype(np.float32)
    # Kümülatif toplam ile brownian noise yaklaşımı
    brown = np.cumsum(white).astype(np.float32)
    brown = brown - np.mean(brown)
    brown_std = np.std(brown) + 1e-8
    brown = (brown / brown_std).astype(np.float32)

    # 3. Birleşik Gürültü
    ambient = hum * 0.6 + brown * 0.4

    # İstenen SNR oranına göre ölçekleme
    if ref_signal is not None and len(ref_signal) > 0:
        sig_power = np.mean(ref_signal ** 2)
        if sig_power > 1e-12:
            target_noise_power = sig_power / (10.0 ** (snr_target_db / 10.0))
            ambient_std = np.std(ambient) + 1e-8
            ambient = ambient * np.sqrt(target_noise_power) / ambient_std

    return ambient.astype(np.float32)


def random_cavity_modes(rng, n_modes=3):
    """Rastgele boşluk rezonansı modları: (frekans_hz, kazanç_db, q)."""
    freqs = np.sort(rng.uniform(80.0, 450.0, n_modes))
    return tuple((float(f), float(rng.uniform(2.0, 9.0)), float(rng.uniform(2.0, 6.0)))
                 for f in freqs)


def apply_rubble_acoustics(y, sr, distance_m=2.0, void_resonance=True, rir_conv=True,
                           noise_snr_db=None, rng=None, modes=None):
    """
    Tam Fiziksel Enkaz Akustiği Boru Hattı:
    Girdi sinyaline sırasıyla:
      1. Yaşam üçgeni boşluk rezonansı (Cavity resonance)
      2. Moloz zayıflaması (Stokes-Kirchhoff viskoelastik sönümleme)
      3. RIR konvolüsyonu (Moloz içi saçılma yankısı)
      4. Saha gürültüsü enjeksiyonu (İsteğe bağlı SNR)
    uygular.
    """
    if len(y) == 0:
        return y.copy().astype(np.float32)

    out = y.copy().astype(np.float32)

    # 1. Boşluk Rezonans Modları
    if void_resonance:
        out = apply_cavity_resonance(out, sr, modes) if modes else apply_cavity_resonance(out, sr)

    # 2. Viskoelastik Sönümleme
    out = viscoelastic_damping(out, sr, distance_m=distance_m)

    # 3. RIR Konvolüsyonu (nedensel: "same" modu sinyali IR uzunluğunun
    # yarısı kadar öne kaydırırdı, bu yüzden tam konvolüsyonun başı alınır)
    if rir_conv:
        rir = generate_rubble_rir(sr, distance_m=distance_m, rng=rng)
        out = fftconvolve(out, rir, mode="full")[:len(out)].astype(np.float32)

    # 4. Enkaz Saha Gürültüsü
    if noise_snr_db is not None:
        noise = generate_rubble_ambient_noise(
            sr=sr,
            duration_sec=len(out) / sr,
            snr_target_db=noise_snr_db,
            ref_signal=out,
            rng=rng
        )
        if len(noise) == len(out):
            out = out + noise

    return np.nan_to_num(out, nan=0.0).astype(np.float32)
