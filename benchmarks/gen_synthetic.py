"""Seeded synthetic embedding-vector generator for batched-birch benchmarks.

Standalone (numpy only). Every distribution returns unit-normalized
float32 vectors plus ground-truth labels (int64; -1 = no ground truth).

Distributions
-------------
uniform   : uniform on the unit sphere (Gaussian rows, normalized).
            No cluster structure; labels are all -1.
blobs     : Gaussian blobs around random sphere centers:
            x = center + (spread / sqrt(dim)) * N(0, I), then
            unit-normalized. ``spread`` is the noise NORM relative
            to the unit center (dimension-free): expected same-blob
            cosine ~= 1 / (1 + spread^2), so spread=0.8 gives ~0.61.
            Labels are blob ids (0..n_blobs-1), balanced round-robin
            assignment shuffled by the seeded RNG.
duplicates: a pool of unique sphere vectors; each row samples one
            pool entry (with replacement). Labels are pool ids, so
            exact duplicates share a label. ``dup_frac`` is the
            expected fraction of rows that duplicate an earlier row:
            pool_size = round(n * (1 - dup_frac)) (at least 1).

CLI (writes <out>.npy, <out>.labels.npy, <out>.config.json):
    python gen_synthetic.py --dist blobs --n 10000 --dim 384 \
        --seed 20261125 --n-blobs 50 --spread 0.35 --out syn-blobs-10k

Also importable: ``from gen_synthetic import generate``.
Large n is generated in blocks to bound peak memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BLOCK = 50_000


def _normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def generate(dist: str, n: int, dim: int, seed: int,
             n_blobs: int = 50, spread: float = 0.8,
             dup_frac: float = 0.5):
    """Return (X float32 unit rows, labels int64, config dict)."""
    if n < 1 or dim < 1:
        raise ValueError('n and dim must be >= 1')
    rng = np.random.default_rng(seed)
    cfg = {'dist': dist, 'n': int(n), 'dim': int(dim), 'seed': int(seed),
           'n_blobs': int(n_blobs), 'spread': float(spread),
           'dup_frac': float(dup_frac)}
    if dist == 'uniform':
        X = np.empty((n, dim), dtype=np.float32)
        for s in range(0, n, BLOCK):
            e = min(n, s + BLOCK)
            X[s:e] = _normalize(rng.normal(size=(e - s, dim))).astype(np.float32)
        labels = np.full(n, -1, dtype=np.int64)
    elif dist == 'blobs':
        if n_blobs < 1:
            raise ValueError('n_blobs must be >= 1')
        centers = _normalize(rng.normal(size=(n_blobs, dim)))
        labels = np.arange(n, dtype=np.int64) % n_blobs
        rng.shuffle(labels)
        sigma = spread / np.sqrt(dim)  # spread = noise norm, dim-free
        X = np.empty((n, dim), dtype=np.float32)
        for s in range(0, n, BLOCK):
            e = min(n, s + BLOCK)
            block = centers[labels[s:e]] + sigma * rng.normal(size=(e - s, dim))
            X[s:e] = _normalize(block).astype(np.float32)
    elif dist == 'duplicates':
        if not 0.0 <= dup_frac < 1.0:
            raise ValueError('dup_frac must satisfy 0 <= dup_frac < 1')
        pool_size = max(1, int(round(n * (1.0 - dup_frac))))
        cfg['pool_size'] = int(pool_size)
        pool = _normalize(rng.normal(size=(pool_size, dim))).astype(np.float32)
        labels = rng.integers(0, pool_size, size=n, dtype=np.int64)
        X = pool[labels].copy()
    else:
        raise ValueError(f'unknown dist {dist!r} (uniform|blobs|duplicates)')
    return X, labels, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dist', required=True, choices=['uniform', 'blobs', 'duplicates'])
    ap.add_argument('--n', type=int, required=True)
    ap.add_argument('--dim', type=int, default=384)
    ap.add_argument('--seed', type=int, default=20261125)
    ap.add_argument('--n-blobs', type=int, default=50)
    ap.add_argument('--spread', type=float, default=0.8)
    ap.add_argument('--dup-frac', type=float, default=0.5)
    ap.add_argument('--out', required=True, help='output prefix (no extension)')
    args = ap.parse_args()
    X, labels, cfg = generate(args.dist, args.n, args.dim, args.seed,
                              n_blobs=args.n_blobs, spread=args.spread,
                              dup_frac=args.dup_frac)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix('.npy'), X)
    np.save(Path(str(out) + '.labels.npy'), labels)
    Path(str(out) + '.config.json').write_text(json.dumps(cfg, indent=2))
    print(f'wrote {out}.npy {X.shape} {X.dtype}, labels {labels.shape}, '
          f'config {cfg}')


if __name__ == '__main__':
    main()
