import numpy as np
from .dtw import dtw


def classify(query_features, database, k=1, band_ratio=0.2, mode="mel"):
    if not database:
        raise ValueError("Baza wzorcow jest pusta.")

    feat_key = "features_rms" if mode == "rms" else "features_mel"
    if feat_key not in database[0]:
        feat_key = "features"

    N = len(query_features)
    scored = []
    for entry in database:
        ref = entry[feat_key]
        dist, _, path = dtw(query_features, ref, band_ratio=band_ratio)
        norm = N + len(ref)
        norm_dist = dist / max(norm, 1)
        scored.append((norm_dist, entry))
    scored.sort(key=lambda x: x[0])
    top_k = [{"dist": d, **e} for d, e in scored[:k]]
    predicted_word    = _weighted_vote(top_k, "word")
    predicted_speaker = _weighted_vote(top_k, "speaker_id")
    return {
        "predicted_word": predicted_word,
        "predicted_speaker": predicted_speaker,
        "best_dist": scored[0][0],
        "top_k": top_k,
    }


def _weighted_vote(candidates, field):
    votes = {}
    for c in candidates:
        v = c[field]
        w = 1.0 / (c["dist"] + 1e-9)
        votes[v] = votes.get(v, 0.0) + w
    return max(votes, key=votes.get)
