import numpy as np
from .dtw import dtw


def classify(query_features, database, k = 1):
    if not database:
        raise ValueError("Baza wzorców jest pusta.")

    scored = []
    for entry in database:
        dist, _, _ = dtw(query_features, entry["features"])
        scored.append((dist, entry))

    scored.sort(key=lambda x: x[0])
    top_k = [{"dist": d, **e} for d, e in scored[:k]]

    predicted_word     = _majority(top_k, "word")
    predicted_speaker  = _majority(top_k, "speaker_id")

    return {
        "predicted_word":    predicted_word,
        "predicted_speaker": predicted_speaker,
        "best_dist":         scored[0][0],
        "top_k":             top_k,
    }


def _majority(candidates, field):
    votes: dict = {}
    for c in candidates:
        v = c[field]
        votes[v] = votes.get(v, 0) + 1
    return max(votes, key=votes.get)
