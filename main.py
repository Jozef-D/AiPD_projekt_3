import numpy as np
from src import *
import sys

path_to_file_1 = "../speaker_14/Nieznormalizowane/benzyna_1.wav"
path_to_file_2 = "../speaker_14/Nieznormalizowane/benzyna_2.wav"

if __name__ == "__main__":

    if len(sys.argv) >= 3:
        path1, path2 = sys.argv[1], sys.argv[2]
    elif (path_to_file_1, path_to_file_2) != "":
        path1, path2 = path_to_file_1, path_to_file_2
    else:
        path1 = input("Ścieżka do pliku 1 (.wav): ").strip()
        path2 = input("Ścieżka do pliku 2 (.wav): ").strip()

    print(f"Wczytywanie: {path1}")
    sig1, sr1, ch1, bps1 = read_wav(open(path1, "rb").read())
    print(f"  próbek: {len(sig1)}, fs: {sr1} Hz, kanały: {ch1}, bit: {bps1}")

    print(f"Wczytywanie: {path2}")
    sig2, sr2, ch2, bps2 = read_wav(open(path2, "rb").read())
    print(f"  próbek: {len(sig2)}, fs: {sr2} Hz, kanały: {ch2}, bit: {bps2}")

    X = extract_frames(sig1)
    Y = extract_frames(sig2)
    print(f"\nDługość sekwencji cech: X={len(X)}, Y={len(Y)}")

    dist, D, path = dtw(X, Y)
    print(f"\nOdległość DTW(X, Y) = {dist:.4f}")
    print(f"Długość ścieżki dopasowania: {len(path)} kroków")
    print(f"Pierwsze 5 punktów ścieżki: {path[:5]}")
    print(f"Ostatnie 5 punktów ścieżki: {path[-5:]}")

