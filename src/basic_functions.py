import struct
import numpy as np



def read_wav(file_bytes):
    if file_bytes[:4] != b'RIFF' or file_bytes[8:12] != b'WAVE':
        raise ValueError("To nie jest plik WAV.")

    pos = 12
    sample_rate = 0
    num_channels = 0
    bits_per_sample = 0
    audio_format = 0
    raw_data = None

    while pos < len(file_bytes) - 8:
        chunk_id = file_bytes[pos:pos + 4]
        chunk_size = struct.unpack_from('<I', file_bytes, pos + 4)[0]
        pos += 8

        if chunk_id == b'fmt ':
            audio_format = struct.unpack_from('<H', file_bytes, pos)[0]
            num_channels = struct.unpack_from('<H', file_bytes, pos + 2)[0]
            sample_rate = struct.unpack_from('<I', file_bytes, pos + 4)[0]
            bits_per_sample = struct.unpack_from('<H', file_bytes, pos + 14)[0]
        elif chunk_id == b'data':
            raw_data = file_bytes[pos:pos + chunk_size]

        pos += chunk_size
        if chunk_size % 2 != 0:
            pos += 1

    if raw_data is None:
        raise ValueError("Brak danych audio w pliku.")
    if audio_format != 1:
        raise ValueError("Obsługiwany tylko format PCM.")

    bps = bits_per_sample
    n = len(raw_data) // (bps // 8)

    if bps == 8:
        samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64)
        samples = (samples - 128.0) / 128.0
    elif bps == 16:
        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        samples /= 32768.0
    elif bps == 24:
        samples = np.zeros(n, dtype=np.float64)
        for i in range(n):
            b0, b1, b2 = raw_data[i * 3], raw_data[i * 3 + 1], raw_data[i * 3 + 2]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val >= 0x800000:
                val -= 0x1000000
            samples[i] = val / 8388608.0
    elif bps == 32:
        samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float64)
        samples /= 2147483648.0
    else:
        raise ValueError(f"Nieobsługiwana głębia: {bps} bit")

    if num_channels == 2:
        samples = (samples[0::2] + samples[1::2]) / 2.0
    elif num_channels > 2:
        mono = np.zeros(len(samples) // num_channels)
        for ch in range(num_channels):
            mono += samples[ch::num_channels]
        samples = mono / num_channels

    return samples, sample_rate, num_channels, bits_per_sample


def normalize_signal(signal):
    peak = np.max(np.abs(signal))
    if peak < 1e-9:
        return signal.copy()
    return signal / peak


WINDOW_TYPES = ("rectangular", "hann", "hamming", "blackman")


def make_window(window_type, length):
    n = np.arange(length, dtype=np.float64)
    N = length - 1

    if window_type == "rectangular":
        return np.ones(length, dtype=np.float64)

    elif window_type == "hann":
        # w[n] = 0.5 * (1 − cos(2π n / N))
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / N))

    elif window_type == "hamming":
        # w[n] = 0.54 − 0.46 * cos(2π n / N)
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / N)

    elif window_type == "blackman":
        # w[n] = 0.42 − 0.5*cos(2πn/N) + 0.08*cos(4πn/N)
        return (0.42
                - 0.50 * np.cos(2.0 * np.pi * n / N)
                + 0.08 * np.cos(4.0 * np.pi * n / N))

    else:
        raise ValueError(
            f"Nieznany typ okna: '{window_type}'. "
            f"Dostępne: {WINDOW_TYPES}"
        )



def extract_frames(signal, frame_len=512, hop_len=256):
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    frames = np.array([
        signal[i * hop_len: i * hop_len + frame_len]
        for i in range(n_frames)
    ])
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    return energy


def extract_frames_fft(signal, frame_len = 512, hop_len = 256, window = "hann", n_fft = None,):
    if n_fft is None:
        n_fft = frame_len

    win = make_window(window, frame_len)
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    n_bins = n_fft // 2 + 1
    features = np.zeros((n_frames, n_bins), dtype=np.float64)

    for i in range(n_frames):
        frame = signal[i * hop_len: i * hop_len + frame_len]
        windowed = frame * win
        spectrum = np.fft.rfft(windowed, n=n_fft)
        magnitude = np.abs(spectrum)
        features[i] = np.log1p(magnitude)

    return features



def mean_fft_spectrum(signal, frame_len = 512, hop_len= 256, window = "hann", n_fft = None, sample_rate = 16000,):
    if n_fft is None:
        n_fft = frame_len

    win = make_window(window, frame_len)
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    n_bins = n_fft // 2 + 1
    acc = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_frames):
        frame = signal[i * hop_len: i * hop_len + frame_len]
        spectrum = np.fft.rfft(frame * win, n=n_fft)
        acc += np.abs(spectrum)

    avg_mag = acc / n_frames
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, avg_mag
