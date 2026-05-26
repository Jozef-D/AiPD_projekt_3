import numpy as np
from .dtw import dtw


def classify(query_features, database, k=1, band_ratio=0.2):
    """k-NN po znormalizowanej odleglosci DTW."""
    if not database:
        raise ValueError("Baza wzorcow jest pusta.")
    scored = []
    for entry in database:
        dist, _, path = dtw(query_features, entry["features"], band_ratio=band_ratio)
        norm_dist = dist / max(len(path), 1)
        scored.append((norm_dist, entry))
    scored.sort(key=lambda x: x[0])
    top_k = [{"dist": d, **e} for d, e in scored[:k]]
    predicted_word    = _weighted_vote(top_k, "word")
    predicted_speaker = _weighted_vote(top_k, "speaker_id")
    return {
        "predicted_word":    predicted_word,
        "predicted_speaker": predicted_speaker,
        "best_dist":         scored[0][0],
        "top_k":             top_k,
    }


def _weighted_vote(candidates, field):
    votes = {}
    for c in candidates:
        v = c[field]
        w = 1.0 / (c["dist"] + 1e-9)
        votes[v] = votes.get(v, 0.0) + w
    return max(votes, key=votes.get)
