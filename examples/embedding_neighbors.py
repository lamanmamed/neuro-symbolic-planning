from pathlib import Path

from neuro_symbolic import load_word_embeddings, nearest_words


ROOT = Path(__file__).resolve().parents[1]
vocabulary, embeddings = load_word_embeddings(ROOT / "models" / "word_embeddings.pth")

for word, similarity in nearest_words("man", vocabulary, embeddings, top_k=5):
    print(f"{word:15s} {similarity:.3f}")
