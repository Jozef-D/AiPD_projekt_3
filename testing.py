from src import load_database, classify

db = load_database("Baza_nagran")

# Weź jedno nagranie jako zapytanie (np. pierwsze z take=2)
query = next(r for r in db if r["take"] == 2)

# Usuń je z bazy żeby nie porównywać ze sobą
train = [r for r in db if r is not query]

result = classify(query["features"], train, k=1)

print(f"Prawdziwe słowo  : {query['word']}")
print(f"Rozpoznane słowo : {result['predicted_word']}")
print(f"Prawdziwy speaker: {query['speaker_id']}")
print(f"Rozpoznany speaker: {result['predicted_speaker']}")
print(f"Odległość DTW    : {result['best_dist']:.4f}")
print(f"\nTop 3 najbliższe:")
for r in result['top_k'][:3]:
    print(f"  dist={r['dist']:.4f}  słowo={r['word']}  speaker={r['speaker_id']}")