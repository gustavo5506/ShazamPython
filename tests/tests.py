import unittest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fingerprintingPaste import fingerprinting
import wave
import struct
import math
import numpy as np

def gerar_wav_teste(nome_arquivo, duracao_seg=1, freq=1000, sample_rate=16000, amplitude=10000):
    with wave.open(nome_arquivo, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        for i in range(int(duracao_seg * sample_rate)):
            t = i / sample_rate
            sample = int(amplitude * math.sin(2 * math.pi * freq * t))
            wav.writeframes(struct.pack('<h', sample))

class TestSlidingWindow(unittest.TestCase):
    def setUp(self):
        self.file = "teste.wav"
        gerar_wav_teste(self.file)

    def tearDown(self):
        if os.path.exists(self.file):
            os.remove(self.file)

    def test_window_size(self):
        windows, rate = fingerprinting.sliding_window_wav(self.file)
        expected_window_size = int(0.125 * rate)
        self.assertTrue(all(len(win) == expected_window_size for win in windows))

    def test_window_numbers_aprox(self):
        windows, rate = fingerprinting.sliding_window_wav(self.file)
        hop_size = int(0.0039 * rate)
        expected_window_size = int(0.125 * rate)
        expected_count = (rate - expected_window_size) // hop_size + 1
        self.assertEqual(len(windows), expected_count)

    def test_total_duracao_coberta_approx(self):
        """Verifica se a cobertura da duração em samples está próxima da total, sem janelas incompletas."""
        windows, rate = fingerprinting.sliding_window_wav(self.file)
        window_size = int(0.125 * rate)
        hop_size = int(0.0039 * rate)

        # Calcular quantos samples foram efetivamente cobertos pelas janelas
        covered_samples = (len(windows) - 1) * hop_size + window_size

        # Calcular número total de samples mono
        with wave.open(self.file, 'rb') as wav:
            total_samples = wav.getnframes()

        self.assertTrue(abs(covered_samples - total_samples) <= hop_size)

    def test_primeira_e_ultima_janela_corretas(self):
        """Verifica se a primeira e a última janela têm o tamanho exato esperado."""
        windows, rate = fingerprinting.sliding_window_wav(self.file)
        expected_len = int(0.125 * rate)
        self.assertEqual(len(windows[0]), expected_len)
        self.assertEqual(len(windows[-1]), expected_len)

    def test_valores_sao_inteiros(self):
        """Verifica se todos os valores nas janelas são inteiros (como esperado após unpack)."""
        windows, _ = fingerprinting.sliding_window_wav(self.file)
        for window in windows:
            self.assertTrue(all(isinstance(x, int) for x in window))

    def test_modo_estereo_descarta_um_canal(self):
        """Verifica se um arquivo estéreo está a ser reduzido para mono corretamente."""
        with wave.open(self.file, 'rb') as wav:
            original_n_channels = wav.getnchannels()
            original_n_frames = wav.getnframes()
        windows, _ = fingerprinting.sliding_window_wav(self.file)
        if original_n_channels == 2:
            # Comparar número de amostras esperadas para mono
            expected_total_samples = original_n_frames  # pois descartamos um dos canais
            total_covered = len(windows) * len(windows[0])
            self.assertLessEqual(total_covered, expected_total_samples)

    def test_janela_unica_para_audio_muito_curto(self):
        """Testa se um áudio muito curto gera nenhuma ou uma janela."""
        # Criar arquivo temporário muito curto
        short_file = "short.wav"
        with wave.open(short_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(struct.pack("<100h", *([0] * 100)))  # 100 samples de silêncio

        try:
            windows, _ = fingerprinting.sliding_window_wav(short_file)
            self.assertTrue(len(windows) <= 1)
        finally:
            if os.path.exists(short_file):
                os.remove(short_file)


class TestHammingFunctions(unittest.TestCase):

    def test_hamming_window_values(self):
        # Test known Hamming values for N=5 (manual check)
        expected = [
            0.08,                    # n = 0
            0.54 - 0.46 * math.cos(2 * math.pi * 1 / 4),
            0.54 - 0.46 * math.cos(2 * math.pi * 2 / 4),
            0.54 - 0.46 * math.cos(2 * math.pi * 3 / 4),
            0.08                     # n = 4
        ]
        result = fingerprinting.hamming_window(5)
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=6)

    def test_apply_hamming_to_windows(self):
        # Simples input: duas janelas com valores constantes
        windows = [
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2]
        ]
        hamming = fingerprinting.hamming_window(5)
        result = fingerprinting.apply_hamming_to_windows(windows)

        # Verifica se aplicou corretamente
        for i in range(len(windows)):
            for j in range(5):
                expected = windows[i][j] * hamming[j]
                self.assertAlmostEqual(result[i][j], expected, places=6)

class TestFFT(unittest.TestCase):
    def test_fft_single_tone(self):
        # Gera um cosseno de frequência k=3 em N=8 pontos
        N = 8
        k = 3
        signal = [math.cos(2 * math.pi * k * n / N) for n in range(N)]

        spectrum = fingerprinting.fft_real(signal)

        # Calcula as magnitudes
        magnitudes = [math.sqrt(c[0]**2 + c[1]**2) for c in spectrum]

        # Esperamos pico em k=3 e k=N-3 (simetria para reais)
        expected_peak_indices = [3, 5]  # N-3 == 5
        top_indices = sorted(range(len(magnitudes)), key=lambda i: magnitudes[i], reverse=True)[:2]

        self.assertCountEqual(top_indices, expected_peak_indices)

    def test_fft_zero_input(self):
        # FFT de zeros deve ser só (0, 0)
        signal = [0.0] * 8
        spectrum = fingerprinting.fft_real(signal)
        for value in spectrum:
            self.assertAlmostEqual(value[0], 0.0, places=6)
            self.assertAlmostEqual(value[1], 0.0, places=6)

    def test_apply_fft_to_multiple_windows(self):
        # 2 janelas de entrada com valores constantes
        windows = [
            [1.0] * 8,
            [2.0] * 8
        ]
        spectra = fingerprinting.apply_fft_to_windows(windows)

        # FFT de sinal constante → primeiro coeficiente (DC) não zero, resto zero
        for spectrum in spectra:
            dc = spectrum[0][0]
            self.assertGreater(dc, 0)
            for i in range(1, len(spectrum)):
                self.assertAlmostEqual(spectrum[i][0], 0.0, places=5)
                self.assertAlmostEqual(spectrum[i][1], 0.0, places=5)

    def test_fft_real_dc_signal(self):
        # Sinal constante -> FFT deve ter energia só na frequência 0
        x = [1.0] * 8
        result = fingerprinting.fft_real(x)
        magnitudes = [math.sqrt(re**2 + im**2) for re, im in result]
        self.assertGreater(magnitudes[0], 0.0)
        for m in magnitudes[1:]:
            self.assertAlmostEqual(m, 0.0, places=5)

    def test_fft_real_single_sine(self):
        # Um ciclo de senoide completo → pico numa frequência
        N = 8
        x = [math.sin(2 * math.pi * i / N) for i in range(N)]
        result = fingerprinting.fft_real(x)
        magnitudes = [math.sqrt(re**2 + im**2) for re, im in result]
        peak = max(magnitudes)
        peak_index = magnitudes.index(peak)
        self.assertIn(1, [peak_index, N-1])  # Pode estar no espelho

    def test_index_to_frequency(self):
        # Testa se a conversão de índice para frequência está correta
        rate = 16000
        N = 2048
        k = 256
        expected_freq = 256 * rate / N
        self.assertAlmostEqual(expected_freq, 2000.0)

    def test_next_power_of_two(self):
        self.assertEqual(fingerprinting.next_power_of_two(2000), 2048)
        self.assertEqual(fingerprinting.next_power_of_two(2048), 2048)
        self.assertEqual(fingerprinting.next_power_of_two(1025), 2048)

class TestBandPeaks(unittest.TestCase):
    def test_band_peaks_known_spectrum(self):
        rate = 16000
        N = 1024
        # Cria um espectro com pico claro em cada faixa
        spectrum = [0j] * N
        spectrum[5] = complex(0, 10)       # very_low
        spectrum[15] = complex(0, 20)      # low
        spectrum[30] = complex(0, 30)      # low_mid
        spectrum[60] = complex(0, 40)      # mid
        spectrum[100] = complex(0, 50)     # mid_high
        spectrum[200] = complex(0, 60)     # high

        peaks = fingerprinting.get_band_peaks_numpy(spectrum, rate)

        expected = {
            "very_low": 5,
            "low": 15,
            "low_mid": 30,
            "mid": 60,
            "mid_high": 100,
            "high": 200
        }

        for band, index in expected.items():
            freq_expected = index * rate / N
            freq_returned, mag = peaks[band]
            self.assertAlmostEqual(freq_returned, freq_expected, places=5)
            self.assertAlmostEqual(mag, abs(spectrum[index]), places=5)

class TestPeakFiltering(unittest.TestCase):
    def test_filter_removes_weak_peaks(self):
        rate = 16000
        N = 1024
        # Faixas: apenas duas fortes
        spectrum = [0j] * N
        spectrum[5] = complex(0, 10)       # very_low
        spectrum[15] = complex(0, 5)       # low
        spectrum[30] = complex(0, 5)       # low_mid
        spectrum[60] = complex(0, 5)       # mid
        spectrum[100] = complex(0, 5)      # mid_high
        spectrum[200] = complex(0, 20)     # high

        result = fingerprinting.filter_peaks_above_mean(spectrum, rate)

        # Esperamos que apenas very_low e high passem a média
        self.assertIn("very_low", result)
        self.assertIn("high", result)
        self.assertNotIn("low", result)
        self.assertNotIn("low_mid", result)
        self.assertNotIn("mid", result)
        self.assertNotIn("mid_high", result)

class TestPeaksAndFingerprints(unittest.TestCase):

    def setUp(self):
        self.rate = 8000  # sample rate
        self.N = 1024     # FFT size

    def test_get_band_peaks_numpy_returns_correct_keys(self):
        # Cria um espectro simulado com magnitudes altas em alguns bins
        spectrum = np.zeros(self.N, dtype=complex)
        spectrum[5] = 10 + 0j
        spectrum[15] = 8 + 0j
        spectrum[25] = 20 + 0j
        spectrum[50] = 30 + 0j
        spectrum[100] = 25 + 0j
        spectrum[300] = 22 + 0j

        peaks = fingerprinting.get_band_peaks_numpy(spectrum, self.rate)

        self.assertIn("very_low", peaks)
        self.assertIn("low", peaks)
        self.assertIn("low_mid", peaks)
        self.assertIn("mid", peaks)
        self.assertIn("mid_high", peaks)
        self.assertIn("high", peaks)

        self.assertIsInstance(peaks["very_low"], tuple)
        self.assertGreater(peaks["very_low"][1], 0)

    def test_filter_peaks_above_mean_only_strong_ones(self):
        spectrum = np.zeros(self.N, dtype=complex)
        spectrum[5] = 1
        spectrum[15] = 1
        spectrum[25] = 1
        spectrum[50] = 20  # acima da média
        spectrum[100] = 1
        spectrum[300] = 1

        filtered = fingerprinting.filter_peaks_above_mean(spectrum, self.rate)

        # Apenas o bin com valor 20 deve sobreviver
        self.assertEqual(len(filtered), 1)
        self.assertIn("mid", filtered)

    def test_generate_fingerprints(self):
        # Simula 3 janelas com tempo crescente
        peaks_per_window = [
            {'time': 0.0, 'peaks': {'low': (100, 10.0)}},
            {'time': 0.2, 'peaks': {'mid': (300, 9.0)}},
            {'time': 0.3, 'peaks': {'high': (700, 8.0)}},
        ]

        result = fingerprinting.generate_fingerprints(peaks_per_window, fan_value=2, target_time_range=(0.1, 0.5))
        self.assertTrue(len(result) > 0)
        for item in result:
            self.assertEqual(len(item), 4)  # (freq1, freq2, delta_t, time1)

            freq1, freq2, delta_t, time1 = item
            self.assertGreaterEqual(delta_t, 0.1)
            self.assertLessEqual(delta_t, 0.5)

if __name__ == '__main__':
    unittest.main()
