import numpy as np
from collections import defaultdict
from .classifier import classify


def _feat_key(mode):
    return "features_rms" if mode == "rms" else "features_mel"


def evaluate_split(database, k=1, mode="mel"):
    groups = defaultdict(list)
    for r in database:
        groups[(r["speaker_id"], r["word"])].append(r)

    train, test = [], []
    for recs in groups.values():
        recs_sorted = sorted(recs, key=lambda r: r["take"])
        if len(recs_sorted) >= 2:
            train.append(recs_sorted[0])
            test.append(recs_sorted[-1])
        else:
            train.append(recs_sorted[0])

    if not train:
        raise ValueError("Pusta baza referencyjna.")
    if not test:
        raise ValueError("Brak danych testowych.")
    return _run_eval(train, test, k=k, mode=mode,
                     label="split per-speaker (najwczesniejsze take->baza, najpozniejsze->test)")


def evaluate_loo(database, k=1, mode="mel"):
    rw, rs = [], []
    n = len(database)
    fk = _feat_key(mode)
    for i, query in enumerate(database):
        train = [r for j, r in enumerate(database) if j != i]
        pred = classify(query[fk], train, k=k, mode=mode)
        rw.append(pred["predicted_word"]    == query["word"])
        rs.append(pred["predicted_speaker"] == query["speaker_id"])
        if (i + 1) % 50 == 0:
            print(f"  LOO: {i+1}/{n} ...")
    return _format_results(rw, rs, n_test=n, mode=f"leave-one-out (k={k}, cechy={mode})")


def _run_eval(train, test, k, mode, label):
    rw, rs = [], []
    fk = _feat_key(mode)
    for i, query in enumerate(test):
        pred = classify(query[fk], train, k=k, mode=mode)
        rw.append(pred["predicted_word"]    == query["word"])
        rs.append(pred["predicted_speaker"] == query["speaker_id"])
        if (i + 1) % 50 == 0:
            print(f"  eval: {i+1}/{len(test)} ...")
    return _format_results(rw, rs, n_test=len(test), mode=f"{label}, k={k}, cechy={mode}")


def _format_results(word_correct, speaker_correct, n_test, mode):
    word_acc = float(np.mean(word_correct)) * 100
    speaker_acc = float(np.mean(speaker_correct)) * 100
    results = {
        "mode": mode,
        "n_test": n_test,
        "word_accuracy": word_acc,
        "speaker_accuracy": speaker_acc,
        "word_correct": int(np.sum(word_correct)),
        "speaker_correct": int(np.sum(speaker_correct)),
    }
    print(f"\n{'='*60}")
    print(f"Ewaluacja: {mode}")
    print(f"{'='*60}")
    print(f"Liczba nagran testowych : {n_test}")
    print(f"Rozpoznawanie slow      : {results['word_correct']}/{n_test} = {word_acc:.1f}%")
    print(f"Identyfikacja osoby     : {results['speaker_correct']}/{n_test} = {speaker_acc:.1f}%")
    print(f"{'='*60}\n")
    return results
