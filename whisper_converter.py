"""
whisper_converter.py - Fısıltı ve İnleme Sentezleme & Ses Dönüşüm Motoru (DSP).

Enkaz altında vokal kordlarını titreştiremeyen kazazedelerin çıkardığı:
1. Fısıltı (Whisper): Vokal kord titreşimi (phonation) yoktur; glottal darbe yerine
   türbülanslı hava akışı (beyaz gürültü) vokal kanal şeklinden geçerek filtrelenir.
   LPC (Linear Predictive Coding) ile sesin formik tınısı korunurken gürültü uyarımı yapılır.
2. İnleme (Moan): Düşük F0 frekansı (70-130 Hz), zayıf harmonikler ve enerjinin 0-500 Hz
   arasına yığıldığı kapalı ağız vokalizasyonu ("hmmm / ıııh").
"""
import numpy as np
import librosa
from scipy.signal import lfilter, butter, sosfilt


def convert_to_whisper_dsp(y, sr=16000, lpc_order=16, whisper_gain=0.15):
    """
    Normal ses kaydını LPC analizi ve aperiyodik beyaz gürültü uyarımıyla
    fısıltı (whisper) sesine dönüştürür.

    Parametreler:
        y (np.ndarray): Giriş ses sinyali
        sr (int): Örnekleme frekansı
        lpc_order (int): LPC kutup sayısı (16 kHz için genelde 16 idealdir)
        whisper_gain (float): Fısıltı ses şiddet çarpanı
    Dönüş:
        np.ndarray: Fısıltı ses sinyali (float32)
    """
    if len(y) == 0:
        return y.copy().astype(np.float32)

    # 1. Pre-emphasis filtresi: Yüksek frekansları parlat
    pre_emphasis = 0.97
    y_pre = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    frame_length = int(0.025 * sr)  # 25 ms çerçeve boyutu
    hop_length = int(0.010 * sr)    # 10 ms atlama boyutu

    if len(y_pre) < frame_length:
        # Ses çok kısaysa doğrudan şekillendirilmiş gürültü dön
        noise = np.random.normal(0, 0.05, len(y)).astype(np.float32)
        return (noise * whisper_gain).astype(np.float32)

    frames = librosa.util.frame(y_pre, frame_length=frame_length, hop_length=hop_length)
    n_frames = frames.shape[1]

    whisper_out = np.zeros(len(y) + frame_length, dtype=np.float32)
    window = np.hamming(frame_length).astype(np.float32)

    # 2. Her çerçeve için all-pole LPC filtresi çıkar ve beyaz gürültüyle uyar
    for i in range(n_frames):
        frame = frames[:, i] * window
        power = np.sum(frame ** 2)
        if power < 1e-8:
            continue

        try:
            a = librosa.lpc(frame, order=lpc_order)
            if np.any(np.isnan(a)) or np.any(np.isinf(a)):
                continue
        except Exception:
            continue

        # Glottal periyodik atım yerine aperiyodik Gauss gürültüsü
        noise_source = np.random.normal(0, 1.0, frame_length).astype(np.float32)
        try:
            raw_filtered = lfilter([1.0], a, noise_source)
            raw_filtered = np.clip(raw_filtered, -50.0, 50.0)
            whisper_frame = raw_filtered.astype(np.float32)
        except Exception:
            continue

        start_sample = i * hop_length
        end_sample = start_sample + frame_length
        whisper_out[start_sample:end_sample] += whisper_frame * window

    whisper_out = whisper_out[:len(y)]

    # 3. Orijinal sinyalin enerji zarfını (RMS) takip et ama fısıltı seviyesine ölçekle
    orig_rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(orig_rms) > 0 and np.max(orig_rms) > 1e-6:
        interp_rms = np.interp(
            np.arange(len(y)),
            np.arange(len(orig_rms)) * hop_length,
            orig_rms
        ).astype(np.float32)
        out_std = np.std(whisper_out) + 1e-8
        whisper_out = whisper_out * (interp_rms * whisper_gain) / out_std
    else:
        whisper_out = whisper_out * whisper_gain

    return np.nan_to_num(whisper_out, nan=0.0).astype(np.float32)


def convert_to_moan_dsp(y, sr=16000, pitch_steps=-6, moan_gain=0.35):
    """
    Normal ses kaydını derin inleme (moaning) akustik profiline dönüştürür:
    - Pitch shift: Ses frekansını 70-130 Hz bandına doğru kaydırır.
    - Low-pass filtre: 450 Hz üzerini söndürür (kapalı dudak/ağız rezonansı).
    - Titreşim dengesizliği (jitter simülasyonu): Halsiz ses telleri.
    """
    if len(y) == 0:
        return y.copy().astype(np.float32)

    # 1. Pitch shift ile frekansı aşağı çek
    try:
        y_low = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
    except Exception:
        y_low = y.copy()

    # 2. 450 Hz Alçak Geçiren Filtre (Butterworth 4. Derece)
    sos = butter(4, 450, btype="low", fs=sr, output="sos")
    y_filtered = sosfilt(sos, y_low).astype(np.float32)

    # 3. Nefes düzensizliği / Tremolo modülasyonu (3-5 Hz yavaş inleme ritmi)
    t = np.arange(len(y)) / sr
    modulator = 0.75 + 0.25 * np.sin(2 * np.pi * 3.5 * t).astype(np.float32)
    y_moan = y_filtered * modulator * moan_gain

    return np.nan_to_num(y_moan, nan=0.0).astype(np.float32)
