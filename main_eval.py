import sys
from src import load_database, evaluate_split, evaluate_loo

DATABASE_ROOT = "Baza_nagran"

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) >= 2 else DATABASE_ROOT

    print(f"Ładowanie bazy: {root}")
    db = load_database(root)

    if not db:
        print("Baza jest pusta – sprawdź ścieżkę i strukturę folderów.")
        sys.exit(1)

    # ── Ewaluacja 1: split take1/take2 ─────────────────────────────
    evaluate_split(db, k=1)

    # ── Ewaluacja 2: leave-one-out (wolniejsza, dokładniejsza) ──────
    # Odkomentuj jeśli chcesz uruchomić LOO:
    # evaluate_loo(db, k=1)

    # ── Opcjonalnie: porównanie k=1 vs k=3 ─────────────────────────
    # evaluate_split(db, k=3)
