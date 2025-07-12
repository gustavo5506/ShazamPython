import wave
import struct
import math
import numpy as np
import os
from Query import init_db, store_fingerprints
import logging

logging.basicConfig(level=logging.DEBUG)

def fingerprinting(wav_path: str) -> dict:
    logging.debug(f"[FP] a abrir {wav_path}")
    windows, rate = sliding_window_wav(wav_path)
    logging.debug(f"[FP] janelas: {len(windows)} @ {rate}Hz")

    hamming_applied = apply_hamming_to_windows(windows)
    logging.debug(f"[FP] hamming OK")

    fft_applied = apply_fft_numpy(hamming_applied)
    logging.debug(f"[FP] FFT feitas: {len(fft_applied)} espectros")

    hop_ms = 3.9
    peaks_per_window = []
    for i, spectrum in enumerate(fft_applied):
        peaks = filter_peaks_above_mean(spectrum, rate)
        peaks_per_window.append({'time': i * hop_ms / 1000, 'peaks': peaks})
    logging.debug(f"[FP] picos extraídos em {len(peaks_per_window)} janelas")

    fp_map = generate_fingerprint_map(peaks_per_window, song_id=os.path.basename(wav_path))
    logging.debug(f"[FP] gerados {len(fp_map)} hashes únicos")
    return fp_map

def sliding_window_wav(file_path, window_ms=125, hop_ms=3.9):
    """
    Reads a WAV file and applies a sliding window of 125ms with a hop of 3.9ms.

    Parameters:
    - file_path: path to the .wav file
    - window_ms: window size in milliseconds
    - hop_ms: step size between windows in milliseconds

    Returns:
    - List of windows (each one is a list of samples)
    - Sampling rate
    """
    with wave.open(file_path, 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        n_frames = wav.getnframes()

        # Read all samples
        frames = wav.readframes(n_frames)

        # Convert bytes to integers
        total_samples = n_frames * n_channels
        if sample_width == 1:
            fmt = f"{total_samples}b"  # 8-bit signed
        elif sample_width == 2:
            fmt = f"<{total_samples}h"  # 16-bit signed little endian
        else:
            raise ValueError("Only 8-bit or 16-bit WAV files are supported.")

        samples = list(struct.unpack(fmt, frames))

        # If stereo, use only one channel
        if n_channels == 2:
            samples = samples[::2]

        # Convert time (ms) to number of samples
        window_size = int((window_ms / 1000) * frame_rate)
        hop_size = int((hop_ms / 1000) * frame_rate)

        # Apply sliding window
        windows = []
        for start in range(0, len(samples) - window_size + 1, hop_size):
            window = samples[start:start + window_size]
            windows.append(window)
        return windows, frame_rate


def hamming_window(N):
    return [0.54 - 0.46 * math.cos((2 * math.pi * n) / (N - 1)) for n in range(N)]


def apply_hamming_to_windows(windows):
    if not windows:
        return []

    window_size = len(windows[0])
    hamming = hamming_window(window_size)

    windowed_segments = []
    for segment in windows:
        windowed = [s * w for s, w in zip(segment, hamming)]
        windowed_segments.append(windowed)
    return windowed_segments


def complex_exp(theta):
    """Returns the complex exponential e^(-j*theta) as a (real, imag) tuple."""
    return (math.cos(theta), -math.sin(theta))

def complex_add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def complex_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def complex_mul(a, b):
    """(a_real, a_imag) * (b_real, b_imag)"""
    return (
        a[0]*b[0] - a[1]*b[1],
        a[0]*b[1] + a[1]*b[0]
    )

def fft_real(x):
    """FFT for real input, returns list of complex tuples (real, imag)."""
    N = len(x)
    if N == 0:
        return [(0.0, 0.0)]
    elif N == 1:
        return [(x[0], 0.0)]

    if N == 2048:
        print("[DEBUG] Starting FFT of size 2048")

    even = fft_real(x[0::2])
    odd = fft_real(x[1::2])

    result = [None] * N
    for k in range(N // 2):
        angle = 2 * math.pi * k / N
        twiddle = complex_exp(angle)
        t = complex_mul(twiddle, odd[k])
        result[k] = complex_add(even[k], t)
        result[k + N // 2] = complex_sub(even[k], t)

    return result

def apply_fft_to_windows(windows):
    spectra = []
    for i, window in enumerate(windows):
        if len(window) == 0:
            print(f"[DEBUG] Empty window at index {i}")
            continue

        # Pad window to next power of 2
        N = len(window)
        N_padded = next_power_of_two(N)
        padded_window = window + [0.0] * (N_padded - N)

        float_win = [float(v) for v in padded_window]
        spectrum = fft_real(float_win)
        spectra.append(spectrum)

        if i % 10 == 0:
            print(f"[INFO] FFT applied to {i} windows...")

    return spectra

def apply_fft_numpy(windows):
    spectra = []
    for window in windows:
        N = len(window)
        N_padded = 1 << (N - 1).bit_length()
        padded = window + [0.0] * (N_padded - N)
        spectra.append(np.fft.fft(padded))
    return spectra


def next_power_of_two(n):
    return 1 << (n - 1).bit_length()

def get_band_peaks2(spectrum, rate):
    # Frequency bands in bin indices
    FREQUENCY_BANDS = {
        "very_low":  (0, 10),
        "low":       (10, 20),
        "low_mid":   (20, 40),
        "mid":       (40, 80),
        "mid_high":  (80, 160),
        "high":      (160, 511)
    }

    N = len(spectrum)
    peaks = {}

    for name, (start, end) in FREQUENCY_BANDS.items():
        max_mag = -1.0
        max_bin = -1

        # Ensure we don't exceed FFT length
        band_end = min(end, N // 2)

        for k in range(start, band_end):
            re, im = spectrum[k]
            magnitude = math.sqrt(re**2 + im**2)

            if magnitude > max_mag:
                max_mag = magnitude
                max_bin = k

        if max_bin != -1:
            freq = max_bin * rate / N
            peaks[name] = (freq, max_mag)
        else:
            peaks[name] = None  # empty band

    return peaks

def get_band_peaks_numpy(spectrum, rate):
    # Frequency bands in bin indices
    FREQUENCY_BANDS = {
        "very_low":  (0, 10),
        "low":       (10, 20),
        "low_mid":   (20, 40),
        "mid":       (40, 80),
        "mid_high":  (80, 160),
        "high":      (160, 511)
    }

    N = len(spectrum)
    peaks = {}

    for name, (start, end) in FREQUENCY_BANDS.items():
        max_mag = -1.0
        max_bin = -1

        band_end = min(end, N // 2)

        for k in range(start, band_end):
            magnitude = abs(spectrum[k])  # directly from complex numbers

            if magnitude > max_mag:
                max_mag = magnitude
                max_bin = k

        if max_bin != -1:
            freq = max_bin * rate / N
            peaks[name] = (freq, max_mag)
        else:
            peaks[name] = None

    return peaks

def filter_peaks_above_mean(spectrum, rate):
    peaks = get_band_peaks_numpy(spectrum, rate)

    # Extract only existing magnitudes
    magnitudes = [mag for (_, mag) in peaks.values() if mag is not None]

    if not magnitudes:
        return {}  # no valid peak

    mean_mag = sum(magnitudes) / len(magnitudes)

    # Filter bands with magnitude >= mean
    filtered = {}
    for band, data in peaks.items():
        if data is None:
            continue
        freq, mag = data
        if mag >= mean_mag:
            filtered[band] = (freq, mag)

    return filtered

def generate_fingerprint_map(peaks_per_window, song_id, fan_value=5, target_time_range=(0.1, 2.0)):
    """
    Generates a fingerprint hashmap where:
    - key: hash_2
    - value: list of (anchor_time, song_id)

    peaks_per_window: list of dictionaries { 'time': float, 'peaks': {band: (freq, mag), ...} }
    song_id: song identifier (string or int)
    """
    fingerprint_map = {}
    n = len(peaks_per_window)

    for i, anchor_window in enumerate(peaks_per_window):
        time1 = anchor_window['time']
        anchor_peaks = anchor_window['peaks']

        for _, (freq1, _) in anchor_peaks.items():
            targets = []
            for j in range(i + 1, n):
                time2 = peaks_per_window[j]['time']
                delta_t = time2 - time1

                if delta_t > target_time_range[1]:
                    break
                if delta_t < target_time_range[0]:
                    continue

                for _, (freq2, _) in peaks_per_window[j]['peaks'].items():
                    targets.append((freq2, delta_t))

            for freq2, delta_t in targets[:fan_value]:
                hash_2 = hash_pair(freq1, freq2, delta_t)
                if hash_2 not in fingerprint_map:
                    fingerprint_map[hash_2] = []
                fingerprint_map[hash_2].append((time1, song_id))

    return fingerprint_map


def hash_pair(freq1, freq2, delta_t, fanout_bits=(10, 10, 12)):
    """
    Creates a unique integer hash from:
    - freq1: anchor frequency
    - freq2: target frequency
    - delta_t: time difference in seconds (converted to ms or quantized)

    fanout_bits: bit allocation for quantization
        - default: 10 bits for freq1, 10 bits for freq2, 12 bits for delta_t
    """
    # Quantization (simple rounding)
    f1 = int(freq1)
    f2 = int(freq2)
    dt = int(delta_t * 1000)  # convert to milliseconds

    # Masks to restrict values to allocated bits
    max_f1 = (1 << fanout_bits[0]) - 1
    max_f2 = (1 << fanout_bits[1]) - 1
    max_dt = (1 << fanout_bits[2]) - 1

    f1 = min(f1, max_f1)
    f2 = min(f2, max_f2)
    dt = min(dt, max_dt)

    # Combine the three values into a single integer
    hash_value = (f1 << (fanout_bits[1] + fanout_bits[2])) | (f2 << fanout_bits[2]) | dt
    return hash_value
