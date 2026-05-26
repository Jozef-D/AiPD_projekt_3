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
        raise ValueError("Obslugiwany tylko format PCM.")

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
            b0, b1, b2 = raw_data[i*3], raw_data[i*3+1], raw_data[i*3+2]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val >= 0x800000:
                val -= 0x1000000
            samples[i] = val / 8388608.0
    elif bps == 32:
        samples = np.frombuffer(raw_data, dtype=np.int32).astype(np.float64)
        samples /= 2147483648.0
    else:
        raise ValueError(f"Nieobslugiwana glebia: {bps} bit")

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
        return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / N))
    elif window_type == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / N)
    elif window_type == "blackman":
        return (0.42
                - 0.50 * np.cos(2.0 * np.pi * n / N)
                + 0.08 * np.cos(4.0 * np.pi * n / N))
    else:
        raise ValueError(f"Nieznany typ okna: {window_type}. Dostepne: {WINDOW_TYPES}")


def extract_frames(signal, frame_len=512, hop_len=256):
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    frames = np.array([
        signal[i*hop_len: i*hop_len + frame_len]
        for i in range(n_frames)
    ])
    energy = np.sqrt(np.mean(frames ** 2, axis=1))
    return energy


def extract_frames_fft(signal, frame_len=512, hop_len=256, window="hann", n_fft=None):
    if n_fft is None:
        n_fft = frame_len
    win = make_window(window, frame_len)
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    n_bins = n_fft // 2 + 1
    features = np.zeros((n_frames, n_bins), dtype=np.float64)
    for i in range(n_frames):
        frame = signal[i*hop_len: i*hop_len + frame_len]
        spectrum = np.fft.rfft(frame * win, n=n_fft)
        features[i] = np.log1p(np.abs(spectrum))
    return features


def mean_fft_spectrum(signal, frame_len=512, hop_len=256, window="hann", n_fft=None, sample_rate=16000):
    if n_fft is None:
        n_fft = frame_len
    win = make_window(window, frame_len)
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    n_bins = n_fft // 2 + 1
    acc = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_frames):
        frame = signal[i*hop_len: i*hop_len + frame_len]
        spectrum = np.fft.rfft(frame * win, n=n_fft)
        acc += np.abs(spectrum)
    avg_mag = acc / n_frames
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    return freqs, avg_mag


def trim_silence(signal, frame_len=512, hop_len=256, threshold_db=30.0):
    if len(signal) < frame_len:
        return signal
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = signal[i*hop_len: i*hop_len + frame_len]
        rms[i] = np.sqrt(np.mean(frame ** 2))
    if rms.max() < 1e-9:
        return signal
    db = 20.0 * np.log10(rms / rms.max() + 1e-12)
    voiced = db > -threshold_db
    if not voiced.any():
        return signal
    first = int(np.argmax(voiced))
    last = len(voiced) - 1 - int(np.argmax(voiced[::-1]))
    start = first * hop_len
    end = min(last * hop_len + frame_len, len(signal))
    return signal[start:end]


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_fft, n_mels, sample_rate, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sample_rate / 2.0
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        left, center, right = bin_pts[m], bin_pts[m+1], bin_pts[m+2]
        if center == left or right == center:
            continue
        for k in range(left, center):
            fb[m, k] = (k - left) / (center - left)
        for k in range(center, right):
            fb[m, k] = (right - k) / (right - center)
    return fb


def extract_features(signal,
                     sample_rate=16000,
                     frame_len=512,
                     hop_len=256,
                     window="hann",
                     n_mels=24,
                     pre_emphasis=0.97,
                     do_normalize=True,
                     do_trim=True):
    """Kanoniczne cechy do klasyfikacji: normalize -> trim -> pre-emphasis ->
    framing -> window -> rFFT -> mel filterbank -> log -> CMN."""
    if do_normalize:
        signal = normalize_signal(signal)
    if do_trim:
        signal = trim_silence(signal, frame_len=frame_len, hop_len=hop_len)
    if pre_emphasis and len(signal) > 1:
        signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))

    win = make_window(window, frame_len)
    n_fft = frame_len
    n_frames = 1 + (len(signal) - frame_len) // hop_len
    fb = mel_filterbank(n_fft, n_mels, sample_rate)
    features = np.zeros((n_frames, n_mels), dtype=np.float64)
    for i in range(n_frames):
        frame = signal[i*hop_len: i*hop_len + frame_len] * win
        spectrum = np.fft.rfft(frame, n=n_fft)
        power = spectrum.real ** 2 + spectrum.imag ** 2
        mel_energy = fb @ power
        features[i] = np.log(mel_energy + 1e-10)
    # Cepstral mean normalization
    features = features - features.mean(axis=0, keepdims=True)
    return features
