import os
import unicodedata
import numpy as np
from .basic_functions import read_wav, extract_features

DIGIT_TO_POLISH = {
    "0": "zero", "1": "jeden", "2": "dwa", "3": "trzy", "4": "cztery",
    "5": "piec", "6": "szesc", "7": "siedem", "8": "osiem", "9": "dziewiec",
    "10": "dziesiec",
}

_POLISH_FOLD = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
    "ó": "o", "ś": "s", "ź": "z", "ż": "z",
    "Ą": "a", "Ć": "c", "Ę": "e", "Ł": "l", "Ń": "n",
    "Ó": "o", "Ś": "s", "Ź": "z", "Ż": "z",
})


def canonical_word(name):
    s = name.lower().strip()
    s = s.translate(_POLISH_FOLD)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.encode("ascii", errors="ignore").decode("ascii")
    s = s.strip()
    return DIGIT_TO_POLISH.get(s, s)


def load_database(root, verbose=True):
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

        gender = parts[2].lower() if len(parts) >= 3 else "?"

        norm_path = os.path.join(speaker_path, "Znormalizowane")
        if not os.path.isdir(norm_path):
            continue

        for fname in sorted(os.listdir(norm_path)):
            if not fname.lower().endswith(".wav"):
                continue
            name = fname[:-4]
            if "_" not in name:
                continue

            last_underscore = name.rfind("_")
            raw_word = name[:last_underscore]
            take_str = name[last_underscore + 1:]
            try:
                take = int(take_str)
            except ValueError:
                continue

            word = canonical_word(raw_word)
            if not word:
                continue

            wav_path = os.path.join(norm_path, fname)
            try:
                with open(wav_path, "rb") as f:
                    raw = f.read()
                sig, sr, ch, bps = read_wav(raw)
                features = extract_features(sig, sample_rate=sr)
            except Exception as e:
                if verbose:
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

    if verbose:
        print(f"Załadowano {len(records)} nagrań "
              f"({len({r['speaker_id'] for r in records})} speakerów, "
              f"{len({r['word'] for r in records})} unikalnych słów).")
    return records


WORDS = set()
MAX_SPEAKER = 999
