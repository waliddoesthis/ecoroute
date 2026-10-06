"""Prompt embeddings, cached to disk so each encoder runs once per prompt set."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_ENCODER = "BAAI/bge-small-en-v1.5"


def embed_texts(
    texts: list[str],
    encoder: str = DEFAULT_ENCODER,
    batch_size: int = 128,
    device: str | None = None,
) -> np.ndarray:
    """L2-normalised embeddings, shape (len(texts), dim). Uses a GPU when one is present."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(encoder, device=device)
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def cached_embeddings(
    prompt_ids: list[str],
    texts: list[str],
    cache: Path,
    encoder: str = DEFAULT_ENCODER,
) -> np.ndarray:
    """Embed only the prompts missing from the cache file, then return all rows in order."""
    known: dict[str, np.ndarray] = {}
    if cache.exists():
        data = np.load(cache, allow_pickle=False)
        if str(data["encoder"]) == encoder:
            known = dict(zip(data["ids"].tolist(), data["vectors"]))
    todo = [i for i, pid in enumerate(prompt_ids) if pid not in known]
    if todo:
        new = embed_texts([texts[i] for i in todo], encoder=encoder)
        known.update({prompt_ids[i]: v for i, v in zip(todo, new)})
        cache.parent.mkdir(parents=True, exist_ok=True)
        ids = list(known)
        np.savez(
            cache,
            ids=np.array(ids),
            vectors=np.stack([known[i] for i in ids]),
            encoder=np.array(encoder),
        )
    return np.stack([known[pid] for pid in prompt_ids])
