"""
sweep_rir.py - Farina Metodu ile Logaritmik Sinüs Taraması ve Enkaz RIR Dekonvolüsyon Motoru.

Bu modül:
1. Angelo Farina'nın logaritmik sinüs taraması (Exponential Sine Sweep - ESS) ve
   ters filtresini (inverse filter) matematiksel olarak üretir.
2. Hoparlör/transdüser ile enkaz içine verilen taramanın mikrofondan kaydedilen
   cevabını ters filtreyle konvolüsyona tabi tutarak (dekonvolüsyon) temiz
   Enkaz İmpuls Yanıtı (Rubble Impulse Response - RIR) çıkarır.
3. Çıkarılan RIR, enkazın tüm yansıma, rezonans ve sönümleme karakteristiklerini
   tek bir wav dosyasında toplar.
"""
import os
import numpy as np
import scipy.signal as signal


def generate_log_sweep_and_inverse(f1=20.0, f2=8000.0, duration_sec=5.0, sr=16000):
    """
    Logaritmik sinüs taraması ve Dirac deltaya eşleşen ters filtresini üretir.

    Parametreler:
        f1 (float): Başlangıç frekansı (Hz)
        f2 (float): Bitiş frekansı (Hz)
        duration_sec (float): Süre (saniye)
        sr (int): Örnekleme hızı (Hz)
    Dönüş:
        sweep (np.ndarray): Tarama sinyali
        inv_filter (np.ndarray): Ters filtre
    """
    total_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False, dtype=np.float64)

    w1 = 2.0 * np.pi * f1
    w2 = 2.0 * np.pi * f2
    R = np.log(w2 / w1)

    # Anlık faz hesaplaması: phi(t) = (w1 * T / R) * (exp(t * R / T) - 1)
    phase = (w1 * duration_sec / R) * (np.exp(t * R / duration_sec) - 1.0)
    sweep = np.sin(phase).astype(np.float64)

    # Ters filtre (Zaman ters çevrilmiş + oktav başına -3 dB genlik eğimi)
    envelope = np.exp(-t * R / duration_sec)
    inv_filter = sweep[::-1] * envelope

    # Dirac delta ölçekleme kalibrasyonu: sweep * inv_filter tepe değeri 1.0 olsun
    test_conv = signal.fftconvolve(sweep, inv_filter, mode="full")
    peak_val = np.max(np.abs(test_conv))
    if peak_val > 1e-12:
        inv_filter /= peak_val

    return sweep.astype(np.float32), inv_filter.astype(np.float32)


def deconvolve_rubble_response(recorded_audio, inv_filter, sr=16000, max_rir_sec=0.5):
    """
    Enkazda kaydedilen sinyal ile ters filtreyi dekonvolüe ederek RIR çıkarır.

    Parametreler:
        recorded_audio (np.ndarray): Enkazdan kaydedilen ses
        inv_filter (np.ndarray): Sweep ters filtresi
        sr (int): Örnekleme frekansı
        max_rir_sec (float): Çıkarılacak maksimum RIR süresi
    Dönüş:
        trimmed_rir (np.ndarray): Temizlenmiş ve tepeye normalize edilmiş RIR
    """
    if len(recorded_audio) == 0:
        return np.zeros(int(sr * max_rir_sec), dtype=np.float32)

    # FFT Konvolüsyonu ile dekonvolüsyon
    raw_rir = signal.fftconvolve(recorded_audio, inv_filter, mode="full")

    # Doğrudan varış (Direct arrival) tepe noktasını bul
    peak_idx = int(np.argmax(np.abs(raw_rir)))

    # Tepe noktasının 5 ms öncesinden başla ve max_rir_sec kadar kuyruğu al
    pre_margin = int(0.005 * sr)
    tail_len = int(max_rir_sec * sr)
    start_idx = max(0, peak_idx - pre_margin)
    end_idx = min(len(raw_rir), start_idx + tail_len)

    trimmed = raw_rir[start_idx:end_idx]

    # Son 30 ms'ye Tukey penceresi ile fade-out uygula (ani kesilme gürültüsünü önler)
    fade_len = min(len(trimmed), int(0.030 * sr))
    if fade_len > 0:
        fade_window = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len)))
        trimmed[-fade_len:] *= fade_window

    # Tepe normalizasyonu
    max_val = np.max(np.abs(trimmed))
    if max_val > 1e-8:
        trimmed = trimmed / max_val

    return trimmed.astype(np.float32)
