# batched-birch 0.1.0 — set-wise BIRCH clustering for embedding vectors

> Draft announcement. Repo: https://github.com/ibielov/batched-birch
> PyPI link pending publication. Nothing here has been posted yet.

Classic BIRCH inserts points one at a time. When your input arrives
in batches — as embedding pipelines produce it — that per-point loop
is the bottleneck. **batched-birch** places a whole wave of items at
once instead: one matmul scores every item against every group
centroid, acceptability and winner selection are vectorized, the
joint CF merge is a segment sum, and the only sequential residue is
splitting overflowed groups. Cluster features are numerically
stable (n, μ, S) with Welford/Chan updates (BETULA-style), so
dispersion does not suffer catastrophic cancellation far from the
origin. NumPy-only core, deterministic (fixed input order, explicit
tie-breaking, no randomness).

```python
import numpy as np
from batched_birch import BirchConfig, BirchTree

tree = BirchTree(BirchConfig(min_similarity=0.5, max_children=16))
tree.fit(X, wave_size=256)   # X: (items, dim), normalized on a copy
labels = tree.labels_
```

## Why set-wise

Same-wave items route against one pinned snapshot of the tree, so a
wave is placed as a unit — a reasonable model for batched ingestion,
and what makes the matrix formulation legal. Items that win no
existing group (including the whole first wave) are consolidated
among themselves set-wise into new groups. `wave_size=1` recovers
classic per-item sequential semantics exactly, at classic cost.
Partitions do depend on wave size; that is documented, measured, and
part of the contract.

This is **not a drop-in replacement** for scikit-learn's `Birch`:
flat groups vs hierarchical CF-tree, cosine gate + dispersion +
capacity vs radius + branching factor, joint wave placement vs
per-point insert. The two do not produce the same partitions.

## Numbers (controlled benchmark, with caveats)

Subprocess arms, imports outside timing, warmups, medians + IQR,
BLAS threads pinned to 1 (both thread settings recorded — pinning is
not neutral: it helps us and hurts sklearn-fit on the benchmark VM).
Full protocol, data, and scripts: https://github.com/ibielov/batched-birch (`benchmarks/REPRODUCING.md`).

- **1,000 WordNet rows: a tie** — 0.058 s vs sklearn-fit 0.055 s.
  (An earlier draft claimed ~13× here; that timed sklearn's import.
  Corrected.)
- **WordNet 31,388 (MiniLM 384-dim):** ours 5.07 s / 4,796 groups /
  coherence 0.504; sklearn-fit (t=0.5) 15.62 s / 20,312 groups;
  sklearn at matched group count (t=0.865) 4.85 s / 5,278 groups /
  coherence 0.373. Different partitions at every setting — read the
  group-shape columns, not just the clock.
- **Synthetic blobs 100k:** ours 43.6 s / 18,824 groups / NMI 0.583
  vs sklearn-fit 211 s / 94,303 groups (91% singletons) / NMI 0.510.
- **Parallel scoring (laptop, 8 cores):** the same 100k blobs run in
  45.9 s (workers=1), 35.1 s (workers=2), 31.5 s (workers=4) —
  identical groups at every worker count. The serial residue caps
  the total gain near 1.8× (Amdahl); BLAS multithreading alone
  measured no gain at this scale.
- **Semantic agreement** with WordNet's own depth-3 hypernym groups:
  NMI 0.485 (ARI ~0, expected at this granularity mismatch). WordNet
  is an informative reference, not ground truth; coherence is not
  correctness — a "plane" cluster (tree / woodworking / airliner /
  math) survives every coherence threshold and is only visible by
  reading. A "more reasonable than WordNet" cluster count exists in
  the repo and is labeled HEURISTIC (mechanical rule + small
  hand-read), not independent review.

## Known limitations

- Flat tree: scoring is O(wave × groups × dim), so fit cost grows
  roughly quadratically in group count. No hierarchical routing yet.
- `BirchTree` retains all vectors as float64 for splits — 3.07 GB of
  history alone at 1M×384. The 1M arm completed on a laptop with
  pre-optimization code (3,769.6 s, 190,225 groups); a rerun on
  current code is pending.
- Wave size changes the partition (4,497 → 4,965 groups across wave
  1 → 1024 on WordNet 31k); coherence is roughly wave-insensitive.
- No persistence/transactions in the core by design — `CFTable` is
  the state your adapter stores.

## Feedback wanted

- Does wave/pinned-snapshot semantics fit your ingestion model, or
  do you need strict per-item ordering?
- Parameter guidance: `min_similarity` / `max_dispersion` /
  `max_children` defaults on non-MiniLM geometries (other dims,
  other embedders).
- Scaling: anyone willing to rerun the 1M arm on current code or
  report >100k behavior on real hardware?
- API shape: `place_wave` / `CFTable` as the adapter surface — what
  is missing?

Install (once published): `pip install batched-birch` ·
Source, benchmarks, and issue tracker:
https://github.com/ibielov/batched-birch · Requires Python
>= 3.10, numpy >= 1.24.
