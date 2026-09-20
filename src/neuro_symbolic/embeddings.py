"""Utilities for loading and inspecting the saved word embeddings."""

from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def load_word_embeddings(checkpoint_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load the vocabulary and 128-dimensional embedding matrix."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vocabulary = checkpoint.get("vocab")
    embeddings = checkpoint.get("embeddings")

    if not isinstance(vocabulary, Sequence) or isinstance(vocabulary, (str, bytes)):
        raise TypeError("Expected checkpoint['vocab'] to be a sequence of words.")
    if embeddings is None:
        raise KeyError("Checkpoint does not contain an 'embeddings' matrix.")

    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if len(vocabulary) != embeddings.shape[0]:
        raise ValueError("Vocabulary size does not match the embedding matrix.")

    return list(vocabulary), embeddings


def nearest_words(
    word: str,
    vocabulary: Sequence[str],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return nearest neighbours by cosine similarity."""
    try:
        index = list(vocabulary).index(word)
    except ValueError as exc:
        raise ValueError(f"'{word}' is not in the vocabulary.") from exc

    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / np.clip(norms, 1e-12, None)
    similarities = normalized @ normalized[index]
    similarities[index] = -np.inf

    order = np.argsort(similarities)[::-1][:top_k]
    return [(str(vocabulary[i]), float(similarities[i])) for i in order]
