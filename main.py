import os
import sys
import numpy as np
from src import read_wav, extract_features, dtw

# Domyślne ścieżki przykładowe (jeśli żadne nie zostaną podane w argumentach):
DEFAULT_1 = "Baza_nagran/speaker_02_f/Znormalizowane/kot_1.wav"
DEFAULT_2 = "Baza_nagran/speaker_02_f/Znormalizowane/kot_2.wav"


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        path1, path2 = sys.argv[1], sys.argv[2]
    elif os.path.isfile(DEFAULT_1) and os.path.isfile(DEFAULT_2):
        path1, path2 = DEFAULT_1, DEFAULT_2
    else:
        path1 = input("Ścieżka do pliku 1 (.wav): ").strip()
        path2 = input("Ścieżka do pliku 2 (.wav): ").strip()

    print(f"Wczytywanie: {path1}")
    with open(path1, "rb") as f:
        sig1, sr1, ch1, bps1 = read_wav(f.read())
    print(f"  próbek: {len(sig1)}, fs: {sr1} Hz, kanały: {ch1}, bit: {bps1}")

    print(f"Wczytywanie: {path2}")
    with open(path2, "rb") as f:
        sig2, sr2, ch2, bps2 = read_wav(f.read())
    print(f"  próbek: {len(sig2)}, fs: {sr2} Hz, kanały: {ch2}, bit: {bps2}")

    X = extract_features(sig1, sample_rate=sr1)
    Y = extract_features(sig2, sample_rate=sr2)
    print(f"\nDługość sekwencji cech: X={X.shape}, Y={Y.shape}")

    dist, D, path = dtw(X, Y)
    norm_dist = dist / max(len(path), 1)
    print(f"\nOdległość DTW(X, Y)               = {dist:.4f}")
    print(f"Znormalizowana (dist/len(path))    = {norm_dist:.4f}")
    print(f"Długość ścieżki dopasowania        = {len(path)} kroków")
    print(f"Pierwsze 5 punktów ścieżki: {path[:5]}")
    print(f"Ostatnie 5 punktów ścieżki: {path[-5:]}")
