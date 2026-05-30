"""Embedding-quality diagnostics shared by the notebook and analysis scripts.

Computes four headline metrics for word2vec embeddings — anisotropy,
Wang-Isola uniformity, top-PC variance ratios, and a within/between
cosine ratio over a user-supplied set of "highlight" words.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from mo_net.samples.word2vec.vocab import Vocab

DEFAULT_HIGHLIGHT_WORDS: Final[tuple[str, ...]] = (
    "king",
    "queen",
    "man",
    "woman",
    "boy",
    "girl",
    "father",
    "mother",
    "son",
    "daughter",
    "city",
    "country",
    "river",
    "mountain",
    "forest",
    "tree",
    "house",
    "car",
    "road",
    "water",
    "fire",
    "earth",
    "wind",
    "sun",
    "moon",
    "star",
    "sea",
    "ocean",
    "rain",
    "snow",
    "book",
    "paper",
    "letter",
    "word",
    "language",
    "music",
    "song",
    "dance",
    "art",
    "science",
    "love",
    "peace",
    "war",
    "power",
    "money",
    "food",
    "bread",
    "wine",
    "animal",
    "dog",
    "cat",
)


@dataclass(frozen=True, kw_only=True)
class HealthMetrics:
    """Computed health metrics for one set of embeddings.

    Cosine-based scalars are in [-1, 1]; variance ratios are in [0, 1];
    Wang-Isola uniformity is negative (more negative = better).
    """

    n_embeddings: int
    anisotropy: float
    uniformity: float
    top1_pc_variance: float
    top3_pc_variance: float
    n_highlight_matched: int
    within_highlight_cosine: float | None
    between_cosine: float | None
    within_between_ratio: float | None


def compute_health_metrics(
    embeddings: np.ndarray,
    vocab: Vocab,
    highlight_words: Sequence[str] = DEFAULT_HIGHLIGHT_WORDS,
    *,
    n_random_pairs: int = 5000,
    n_background_sample: int = 500,
    seed: int = 42,
) -> HealthMetrics:
    """Compute the four headline embedding-health metrics + a within/between
    cosine ratio for ``highlight_words`` (silently dropped if not in vocab).
    """
    E = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    En = E / norms

    rng = np.random.default_rng(seed)
    N = len(En)
    n_pairs = min(n_random_pairs, N * (N - 1) // 2)
    # Oversample to compensate for the i==j drops, then truncate.
    i_idx = rng.integers(0, N, n_pairs * 2)
    j_idx = rng.integers(0, N, n_pairs * 2)
    keep = i_idx != j_idx
    i_idx, j_idx = i_idx[keep][:n_pairs], j_idx[keep][:n_pairs]
    random_cosines = np.sum(En[i_idx] * En[j_idx], axis=1)
    anisotropy = float(random_cosines.mean())
    # Wang-Isola: log E[exp(-2 * ||x - y||^2)] on unit-normalised pairs.
    # ||x - y||^2 = 2 - 2 cos(x, y) when both are unit vectors.
    sq_dists = 2.0 - 2.0 * random_cosines
    uniformity = float(np.log(np.exp(-2.0 * sq_dists).mean()))

    E_centered = E - E.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(E_centered, full_matrices=False)
    var_ratios = (s**2) / (s**2).sum()
    top1 = float(var_ratios[0])
    top3 = float(var_ratios[:3].sum())

    in_vocab = set(vocab.vocab)
    hl_words = [w for w in highlight_words if w in in_vocab]
    within: float | None = None
    between: float | None = None
    ratio: float | None = None
    if len(hl_words) >= 2:
        hl_ids = np.array([vocab[w] for w in hl_words], dtype=int)
        hl_E = En[hl_ids]
        hl_sim = hl_E @ hl_E.T
        np.fill_diagonal(hl_sim, np.nan)
        within = float(np.nanmean(hl_sim))
        hl_set = set(int(i) for i in hl_ids)
        non_hl = np.array([i for i in range(N) if i not in hl_set], dtype=int)
        bg = rng.choice(
            non_hl, size=min(n_background_sample, len(non_hl)), replace=False
        )
        between = float((hl_E @ En[bg].T).mean())
        if abs(between) > 1e-12:
            ratio = within / between

    return HealthMetrics(
        n_embeddings=N,
        anisotropy=anisotropy,
        uniformity=uniformity,
        top1_pc_variance=top1,
        top3_pc_variance=top3,
        n_highlight_matched=len(hl_words),
        within_highlight_cosine=within,
        between_cosine=between,
        within_between_ratio=ratio,
    )
