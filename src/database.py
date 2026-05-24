import os
import numpy as np
from .basic_functions import read_wav, extract_frames

WORDS = {
    "zero", "jeden", "dwa", "trzy", "cztery",
    "piec", "szesc", "siedem", "osiem", "dziewiec",
    "dziesiec", "kwiat", "pies", "kot", "samochod",
    "jajko", "benzyna", "chrzaszcz", "odrzutowiec", "dzwiek",
    "przetwarzanie",
    # wersje z polskimi znakami (na wypadek gdyby nazwy plików je miały (nie chciało mi się sprawdzać))
    "pięć", "sześć", "dziewięć", "dziesięć",
    "samochód", "chrząszcz", "dźwięk",
}

MAX_SPEAKER = 20


def load_database(root):
    records = []

    if not os.path.isdir(root):
        raise FileNotFoundError(f"Nie znaleziono katalogu: {root}")

    for speaker_dir in sorted(os.listdir(root)):
        speaker_path = os.path.join(root, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue

        parts = speaker_dir.split("_")
        if len(parts) < 2 or parts[0].lower() != "speaker":
            continue

        try:
            speaker_id = int(parts[1])
        except ValueError:
            continue

        if speaker_id > MAX_SPEAKER:
            continue

        gender = parts[2].lower() if len(parts) >= 3 else "?"

        norm_path = os.path.join(speaker_path, "Znormalizowane")
        if not os.path.isdir(norm_path):
            continue

        for fname in sorted(os.listdir(norm_path)):
            if not fname.lower().endswith(".wav"):
                continue

            name = fname[:-4]  # bez .wav
            if "_" not in name:
                continue

            last_underscore = name.rfind("_")
            word = name[:last_underscore]
            take_str = name[last_underscore + 1:]

            try:
                take = int(take_str)
            except ValueError:
                continue

            if word not in WORDS:
                continue

            wav_path = os.path.join(norm_path, fname)
            try:
                raw = open(wav_path, "rb").read()
                sig, sr, ch, bps = read_wav(raw)
                features = extract_frames(sig)
            except Exception as e:
                print(f"  [SKIP] {wav_path}: {e}")
                continue

            records.append({
                "speaker_id": speaker_id,
                "gender":     gender,
                "word":       word,
                "take":       take,
                "path":       wav_path,
                "features":   features,
            })

    print(f"Załadowano {len(records)} nagrań "
          f"({len({r['speaker_id'] for r in records})} speakerów, "
          f"{len({r['word'] for r in records})} słów).")
    return records
