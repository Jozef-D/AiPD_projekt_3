import numpy as np
from .classifier import classify



def evaluate_split(database, k = 1):
    train = [r for r in database if r["take"] == 1]
    test  = [r for r in database if r["take"] == 2]

    if not train:
        raise ValueError("Brak nagrań z take=1 w bazie.")
    if not test:
        raise ValueError("Brak nagrań z take=2 w bazie.")

    return _run_eval(train, test, k=k, mode="split (take1→baza, take2→test)")



def evaluate_loo(database, k = 1):
    results_word     = []
    results_speaker  = []
    n = len(database)

    for i, query in enumerate(database):
        train = [r for j, r in enumerate(database) if j != i]
        pred = classify(query["features"], train, k=k)
        results_word.append(pred["predicted_word"]    == query["word"])
        results_speaker.append(pred["predicted_speaker"] == query["speaker_id"])

        if (i + 1) % 50 == 0:
            print(f"  LOO: {i+1}/{n} ...")

    return _format_results(results_word, results_speaker,
                           n_test=n, mode=f"leave-one-out (k={k})")


def _run_eval(train, test, k, mode):
    results_word = []
    results_speaker = []

    for i, query in enumerate(test):
        pred = classify(query["features"], train, k=k)
        results_word.append(pred["predicted_word"]    == query["word"])
        results_speaker.append(pred["predicted_speaker"] == query["speaker_id"])

    return _format_results(results_word, results_speaker,
                           n_test=len(test), mode=f"{mode}, k={k}")


def _format_results(word_correct, speaker_correct, n_test, mode):
    word_acc = np.mean(word_correct) * 100
    speaker_acc = np.mean(speaker_correct) * 100

    results = {
        "mode": mode,
        "n_test": n_test,
        "word_accuracy": word_acc,
        "speaker_accuracy": speaker_acc,
        "word_correct": int(np.sum(word_correct)),
        "speaker_correct": int(np.sum(speaker_correct)),
    }

    print(f"\n{'='*50}")
    print(f"Ewaluacja: {mode}")
    print(f"{'='*50}")
    print(f"Liczba nagrań testowych : {n_test}")
    print(f"Rozpoznawanie słów      : {results['word_correct']}/{n_test} = {word_acc:.1f}%")
    print(f"Identyfikacja osoby     : {results['speaker_correct']}/{n_test} = {speaker_acc:.1f}%")
    print(f"{'='*50}\n")

    return results
