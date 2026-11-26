# batched-birch

Set-wise (batched) BIRCH clustering over embedding vectors. NumPy-only core.

Classic BIRCH inserts points one at a time: each point descends the CF
tree, updates statistics along the path, and may trigger a split. That
per-point loop is the bottleneck when input arrives in batches, as
embedding pipelines produce it. This library places a whole *wave* of
items at once with matrix operations: scoring, acceptability, and
winner selection are vectorized, the joint CF merge is a vectorized
Chan merge, and the only sequential residue is splitting overflowed
groups.

The core knows vectors and CF (clustering feature) statistics, nothing
else. Embedders, databases, and persistence are adapters around the
core, never inside it. If you want persistence, you store the
`CFTable` columns; if you want extra scoring signals, you add them
outside `place_wave`.

Requires: Python >= 3.10, numpy >= 1.24. scikit-learn is needed only
for the benchmarks (`pip install -e .[dev]`).

## Quick start

```python
import numpy as np
from batched_birch import BirchConfig, BirchTree

X = np.load("vectors.npy")          # (items, dim); need not be pre-normalized
tree = BirchTree(BirchConfig(min_similarity=0.5, max_children=16))
tree.fit(X, wave_size=256)          # or partial_fit per incoming batch
labels = tree.labels_               # group row index per item
print(tree.n_groups)
```

## How this differs from scikit-learn's `Birch`

This is **not a drop-in replacement** for `sklearn.cluster.Birch`, and
the two do not produce the same partitions. The differences are
structural:

| | batched-birch (this library) | scikit-learn `Birch` |
|---|---|---|
| Tree | Flat list of groups (`parent` column kept for forward compatibility; no internal routing nodes in this version) | Hierarchical CF-tree with internal nodes |
| Insertion | Whole wave placed jointly against one pinned snapshot (see below) | One point at a time, descending the tree |
| Join rule | Cosine gate (`min_similarity`) + after-insert dispersion limit (`max_dispersion`) + leaf capacity (`max_children`) | Euclidean subcluster radius (`threshold`) + `branching_factor` |
| Similarity | Cosine on unit-normalized vectors | Euclidean |

Because the join rules differ, there is no parameter mapping that
makes the two equivalent. Comparisons in this README are speed and
group-shape trade-offs on specific distributions, not identical
partitions at different speeds.

## Wave semantics (read this before choosing a wave size)

Items arrive in waves (default 256). Same-wave items route against
**one pinned snapshot** of the tree — they do not see each other's
assignments to *existing* groups. A wave is scored as a unit, then
merged.

Per wave (`place_wave`):

1. **Score** — one matrix multiply: every item × every group centroid
   (`X @ C.T`). Cosine similarity on unit-normalized vectors. With
   `workers > 1` the centroid rows are split into contiguous blocks
   scored in parallel (one matmul per block, exact same shortlist —
   see "Parallel scoring" below); the full score matrix is then
   never materialized.
2. **Shortlist** — top-`candidate_count` groups per item
   (`argpartition` + deterministic boundary-tie fallback; ties break
   to the lowest group index).
3. **Acceptability** — vectorized over (item, candidate) pairs:
   after-insert CF statistics are computed elementwise; a candidate is
   acceptable if similarity >= `min_similarity`, after-insert
   dispersion <= `max_dispersion`, and group size <= `max_children`.
4. **Winner** — lowest after-dispersion, then highest similarity,
   then lowest group index. Explicit tie-breaking everywhere.
5. **Joint merge** — items landing in the same group merge their CFs
   simultaneously: per-group (count, mean, S) of the new points via
   segment ops, then a vectorized Chan merge with each touched
   group's existing CF.
6. **New groups (within-wave consolidation)** — items that win no
   existing group, including every item of a first wave on an empty
   tree, are consolidated *among themselves* before any new group is
   committed: rounds over the pinned pairwise similarities of the
   unassigned pool form new groups set-wise (largest ranked prefix
   whose joint CF satisfies `max_children` and `max_dispersion`).
   An item similar to nothing stays a singleton. An item that fails
   every existing group cannot join one through a same-wave neighbor
   that succeeded; it can only consolidate with other unassigned
   items.
7. **Splits** — groups that overflow (size or dispersion) after the
   merge are listed in `splits_needed`; `place_wave` itself does not
   split. `BirchTree` resolves them with `split_group` (classic
   farthest-pair rule) until no group violates.

Implications:

- **Partitions depend on wave size.** There is no subset/superset
  relationship between wave sizes. On WordNet 31,388 (below), group
  counts were 4,497 (wave 1) → 4,689 (wave 32) → 4,796 (wave 256) →
  4,965 (wave 1024), while coherence stayed in 0.498–0.506. Pick a
  wave size and keep it fixed if you need stable partitions.
- **`wave_size=1` is sequential semantics**: `fit(X, wave_size=1)`
  reproduces classic per-item insertion (each item sees all previous
  items' groups). It is also the slowest setting by far — 147.8 s vs
  5.07 s at wave 256 on WordNet 31k — because per-wave fixed costs
  then run per item.
- **Determinism.** Fixed input order, no randomness, explicit
  tie-breaking at every decision point. Same input → same tree, on
  any platform. Input trees are never mutated; `place_wave` returns
  the updated tree.

## API

- `BirchConfig` — frozen dataclass with the five knobs below.
  Ranges are validated at construction (`ValueError` on violation).
- `BirchTree(cfg=None, dim=None)` — stateful wrapper.
  - `fit(X, wave_size=256)` — **resets** the model (fresh tree, fresh
    history), then fits `X` in waves. The whole input is validated
    before the reset, so invalid input leaves the previous model
    untouched. Fitting 0 rows yields an empty model.
  - `partial_fit(X)` — one incremental wave (place + split
    resolution). A 0-row input is a no-op that only fixes the
    dimension if none is known yet.
  - `labels_` — group row index per item, in arrival order.
  - `close()` — releases the parallel-scoring thread pool
    (`workers > 1` only; recreated lazily on next use).
  - `n_groups`, `tree` (the `CFTable`), `cfg`.
  - `BirchTree` retains every (normalized) vector it has seen, which
    is needed to gather members for splits. See scaling limits.
- `place_wave(X, tree, cfg) -> WaveResult` — the set-wise core.
  Returns `assignment`, `new_groups` (items assigned to groups
  created this wave), `splits_needed`, the updated `tree`, and
  `deferred`. Does not mutate the input tree and does not split.
- `split_group(tree, group, member_X, cfg)` — farthest-pair split of
  one group; returns `(new_tree, [group, new_index], labels)`. Does
  not mutate the input tree.
- `CFTable` — the whole tree state as typed columns (`n`, `mu`, `S`,
  `parent`, `ids`), one row per group. `centroids()` and
  `dispersion()` are vectorized. This is what an adapter persists.
  `append_singletons` / `append_groups` are raw builders: they store
  vectors exactly as given (validated finite and non-zero) and
  expect unit vectors.

**Cluster features are numerically stable (BETULA-style).** Each
group's CF is the triple (`n`, `mu`, `S`) — count, mean vector, and
sum of squared deviations from the mean — maintained with Welford
point updates and Chan group merges. Dispersion is `S / n`,
computed directly. This replaces the classic BIRCH (N, LS, SS)
form, whose dispersion `SS/N − ‖LS‖²/N²` catastrophically cancels
for data far from the origin (at offset 10⁸ the classic form
returned dispersion 0.0 for data whose true dispersion is ~32; the
stable form is exact). On unit-normalized embeddings the two forms
agree to ~10⁻¹⁵, and the switch was verified equivalence-proven:
byte-identical labels on blobs 1k, WordNet 31k, and blobs 100k
(132,388 items, zero flips), same wall/RSS within noise.

**Configuration** (`BirchConfig`):

| knob | default | meaning | guidance |
|---|---|---|---|
| `min_similarity` | 0.5 | Cosine gate for joining an existing group (and for consolidation rounds). Valid range: `0 < min_similarity <= 1`. | Higher = stricter, more/smaller groups. This is the main granularity knob. |
| `max_dispersion` | 0.5 | Mean squared radius limit (compactness). Must be `> 0`. | Lower = tighter groups, more splits. Interacts with `min_similarity`; tune both on your geometry. |
| `max_children` | 16 | Leaf capacity (max members per group). Must be `>= 1`. | Hard cap; groups at capacity split on overflow. Raise for larger topical groups. |
| `candidate_count` | 5 | Shortlist size per item. Must be `>= 1`. | Larger = more candidates checked per item (slower, occasionally better winner). 5 is usually enough; raise if winners look arbitrary. |
| `workers` | 1 | Parallel scoring blocks. Must be an integer `>= 1`. | `1` = serial path, no threads. `> 1` splits centroid rows into that many blocks scored in parallel; results are identical (see "Parallel scoring"). Pin BLAS to 1 thread per worker or you oversubscribe. |

**Input validation and unit-norm policy.** Inputs are *not* required
to be pre-normalized. Every public entry point (`place_wave`,
`split_group`, `BirchTree.fit` / `partial_fit`) validates its input
and works on a unit-normalized float64 **copy**; the caller's array
is never mutated, and similarity is cosine. Inputs must be 2-D,
finite (no NaN/inf), free of zero rows, and dimension-consistent
with the tree; violations raise `ValueError`. CF arithmetic is
float64 regardless of input dtype.

## Results

Controlled benchmark, 2026-11-25. Full data:
`results/benchmark-2026-11-25.json` (38 arms) and
`results/semantic-evidence-2026-11-25.json`; method and reading in
`benchmarks/REPRODUCING.md`, `benchmarks/SEMANTIC-EVIDENCE.md`, and
`benchmarks/BOTTLENECKS.md`. The older single-run files in `results/`
(`birch-vs-sklearn-*.json`, `wordnet-grouping-comparison-*.json`,
`three-way-*.json`) are the *before* picture for pre-fix code and are
kept untouched; do not expect them to match. In particular, an
earlier version of this README claimed "~13× faster at 1,000 items" —
that was wrong: the old script timed scikit-learn's import. The
controlled numbers below replace it.

**Method in brief.** One subprocess per arm; imports and data loading
outside the timed region (only the fit call is timed; sklearn
`predict` for labels is timed separately and never inside `wall_s`).
One untimed warmup, then timed reps with a fresh model each rep
(3 reps up to 31k rows, 2 at 100k, 1 for diagnostics); reported wall
time is the median, spread is the IQR. Label hashes verify reps are
identical. BLAS threads pinned to 1
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`) unless
an arm says `threads-default`. Environment: Python 3.12.3, NumPy
2.5.5, scikit-learn 1.9.1, AMD EPYC 9D25 (1.5 shared CPUs), 7.7 GB
RAM, Linux. Every fragment records its own environment.

**Thread pinning is not neutral.** On this VM, threads=1 *helps*
batched-birch (5.07 s vs 11.13 s at default threads on WordNet 31k,
oversubscription on 1.5 shared CPUs) and *hurts* sklearn-fit
(15.62 s vs 13.39 s). Both settings are recorded in the JSON. Never
compare a threads=1 number against a default-threads number; on a
real multi-core laptop the sign may flip. At 100k synthetic blobs
on this VM, a direct default-threads run took 385.1 s vs the 43.6 s
pinned median (same 18,824 groups) — oversubscription again; see
`results/threads-default-100k-vm-2026-11-25.json`. Multi-core host
measurement (laptop, 8 cores, 100k blobs): default threads 45.1 s
vs pinned 45.9 s, same 18,824 groups — the sign does not flip at
this scale; see `results/host-results-2026-11-26.json`.

**Enabling multi-threaded BLAS.** The library does not pin threads;
the wave scoring matmul (`X @ C.T`) runs through NumPy's BLAS
(OpenBLAS/MKL). Thread count is controlled by environment variables
read at NumPy import: leave `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` unset (BLAS default:
all cores) or set them to the desired count *before* importing
NumPy. The benchmark script pins them to 1 in its subprocess env
for controlled comparisons; `threads-default` arms skip the pin.
Only BLAS operations parallelize — elementwise NumPy stays
single-threaded — but the matmul dominates at scale. sklearn's
Birch has no equivalent lever: its per-point Python tree descent
has no large matrix operation for BLAS to accelerate.

**Parallel scoring (`workers > 1`, "Mode B").** The alternative to
multithreaded BLAS: split the centroid rows into `workers`
contiguous blocks, score each block in its own thread (NumPy
matmul releases the GIL), take each block's local top-k, and merge
into the exact global top-k (score desc, group index asc — the
same shortlist the serial path produces, ties included; verified
identical labels at 1k/31k/100k for workers 1/2/4). The
partition is over group rows only, never the feature dimension,
so every score is the same dot product — up to ~1 ULP of BLAS
kernel rounding when block shapes change, which does not alter
selection on the tested datasets. Side benefit: the
full wave×groups score matrix is never materialized. `BirchTree`
keeps one persistent thread pool (created lazily; `close()`
releases it); the free `place_wave` uses a pool scoped to the
call. **Do not combine the two modes:** with `workers > 1`, pin
BLAS to one thread per worker (`OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` before importing NumPy).
8 workers × 8 BLAS threads = 64 competing threads and can be
dramatically slower than either mode alone. The Python-level
residue (merge, acceptability, consolidation, splits) stays
serial in both modes, so speedup is bounded by the scoring share.
VM threading numbers do not predict laptop speedup (this VM has
1.5 shared CPUs); the controlled arms `__workers2` / `__workers4`
in `bench_controlled.py` measure it on real hardware. Measured on
a laptop (AMD Ryzen 7 7735HS, 8 cores), synthetic blobs 100k,
BLAS pinned to 1 thread per worker: **workers=1: 45.9 s,
workers=2: 35.1 s (1.31×), workers=4: 31.5 s (1.46×)** — 18,824
groups at every worker count. Scaling flattens (w2→w4: 1.11×):
the parallel scoring fraction is ~45%, so the serial residue caps
total gain near 1.8×. Multithreaded BLAS alone (workers=1,
threads unpinned) measured 45.1 s — no gain at this scale. Full
host record: `results/host-results-2026-11-26.json`.

Arms: `ours` = `BirchTree(BirchConfig()).fit(X, wave_size=256)`
(cosine gate 0.5, dispersion ≤ 0.5, capacity 16); `sklearn-fit` /
`sklearn-partial` = `Birch(threshold=0.5, branching_factor=50,
n_clusters=None)` (threshold is a Euclidean subcluster *radius*, not
a cosine gate; partial_fit in waves of 256); `sklearn-matched` =
sklearn fit at threshold 0.865, tuned so its WordNet-31k group count
approximates ours. Matched granularity is not matched semantics.

### WordNet 31,388 (threads=1, medians)

31,388 Open English WordNet 2025 synsets rendered as sentence-form
texts, embedded with MiniLM (`all-MiniLM-L6-v2`, 384-dim). Row
selection was corpus-driven (private workbench); see Reproducing.

| arm | wall (s) | groups | mean size | singleton items | coherence* | peak RSS |
|---|---|---|---|---|---|---|
| batched-birch w256 | 5.07 | 4,796 | 6.55 | 2.8% | 0.504 | 508 MB |
| sklearn-fit t=0.5 | 15.62 | 20,312 | 1.55 | 42.8% | 0.718 | 764 MB |
| sklearn-partial t=0.5 | 10.23 | 20,312 | 1.55 | 42.8% | 0.718 | 823 MB |
| sklearn-fit t=0.865 (matched granularity) | 4.85 | 5,278 | 5.95 | 6.2% | 0.373 | 407 MB |

\* pair-weighted mean intra-cluster cosine.

At 1,000 rows (WordNet prefix, threads=1) it is a **tie**: ours
0.058 s vs sklearn-fit 0.055 s. At 31k, ours is faster than the
t=0.5 sklearn arms but produces a very different partition (4,796
topical groups vs 20,312 mostly-singleton subclusters); at matched
group count (t=0.865) sklearn-fit is slightly faster (4.85 s) with
lower coherence (0.373 vs 0.504) and much larger max groups (291 vs
16, the latter capped by `max_children`). No arm here supports a
claim of universal superiority — only speed/quality trade-offs on
these distributions.

Group counts dropped vs the pre-fix code (5,209 → 4,796 at 31k)
because of the first-wave consolidation fix: the old code turned the
entire first wave into singletons. At the 1,000-row prefix the change
is larger (591 → 353 groups). Old 1k quality numbers are not
comparable to the current code.

A bookkeeping-only performance fix (capacity-doubling history buffer
instead of a full `vstack` copy per wave) cut 31k wall time by 38%
(7.85 s → 4.86 s in the before/after check) with **byte-identical
labels**; see `benchmarks/BOTTLENECKS.md`.

### Synthetic blobs 100,000 (threads=1, medians of 2 reps)

50 blobs, spread 0.8, dim 384, seed 20261125.

| arm | wall (s) | groups | singleton items | NMI vs ground truth | ARI vs ground truth |
|---|---|---|---|---|---|
| batched-birch w256 | 43.6 | 18,824 | 2.1% | 0.583 | 0.008 |
| sklearn-fit t=0.5 | 211.1 | 94,303 | 91.3% | 0.510 | 0.0002 |
| sklearn-partial t=0.5 | 136.9 | 94,303 | 91.3% | 0.510 | 0.0002 |

The 1M-row arm **completed on a laptop** (AMD Ryzen 7 7735HS,
8 cores) with pre-optimization code: **3,769.6 s, 190,225 groups**
(single rep, BLAS pinned to 1). Both implementations hit
`memory_error` under this VM's 5.5 GB guard (float64 history alone
is 3.07 GB at 1M×384). Since the retention fix below, `fit` keeps
exactly one copy of the data (the normalized input matrix becomes
the history buffer), so the fit peak is one n·d float64 copy lower
than when that arm was recorded; a 1M rerun on current code is
pending. To run the heavy arms on a full laptop:
`python benchmarks/bench_controlled.py --only syn-blobs-1000000 --timeout 14400`
(≥16 GB RAM recommended). Host measurements are recorded in
`results/host-results-2026-11-26.json`.

### Semantic agreement with WordNet (informative, not ground truth)

WordNet's lexicographer files are a lexicographic filing system, not
a topical ground truth: a sailing cluster spanning artifact/motion/
act files disagrees with WordNet and can still be topically right.
Agreement shows the embeddings recover lexicographic structure, not
that clusters are semantically correct. Embeddings give similarity
candidates, not verified concepts. Read NMI, ARI, singleton rate,
size, and coherence together — no single column is "quality."

Our partition (4,796 groups) vs WordNet's own groupings:

| reference | ref groups | NMI | ARI |
|---|---|---|---|
| Lexicographer files | 41 | 0.372 | 0.003 |
| Hypernym ancestors, depth 2 | 1,375 | 0.372 | 0.001 |
| Hypernym ancestors, depth 3 | 3,155 | 0.485 | 0.002 |

Our clusters sit at roughly depth-3 granularity. ARI near zero is
expected at this granularity mismatch (ARI penalizes the 41-vs-4,796
split heavily); NMI is the informative column.

**"More reasonable than WordNet" is a HEURISTIC, not a finding.** A
mechanical rule (cross-lexname clusters with coherence ≥ 0.30 and
within−cross cosine gap < 0.15) plus the author's hand-read of a
small sample classifies 2,915 of 3,915 multi-member clusters as
cross-lexname-coherent vs 475 split-candidates. That is not
independent review and not inter-annotator measured; the thresholds
are choices. Quote the counts only with the HEURISTIC label
attached. See `benchmarks/SEMANTIC-EVIDENCE.md`.

**Coherence is not correctness.** Coherence measures embedding
geometry (mean pairwise cosine). A cluster can be geometrically tight
and semantically wrong. Known failure mode, still present in the
current code — **shared-headword binding** (clusters united by word
*form* across unrelated meanings): the "plane" cluster (11 members:
London plane the tree, carpenter's/match/router plane the tools,
plane seat on an airliner, plane table in surveying, Cartesian and
tangent plane in math) persists. No cluster fell below 0.20
coherence, so coherence thresholds do not detect this failure mode;
only reading does. Any downstream use that needs semantic
correctness needs an audit step, not a coherence threshold.

## Scaling limits

From `benchmarks/BOTTLENECKS.md` (profile at 31k; asymptotics are the
point, absolute seconds will differ on your hardware):

- **Scoring is O(wave × groups × dim)** per wave — one matmul, but
  groups grow with `n`, so total fit cost is O(n · G · d), roughly
  quadratic in the number of groups. There is no hierarchical
  routing in this version (flat tree by design); a CF-tree with
  internal nodes is the known fix and is explicitly out of scope.
- **Memory: O(n · dim) retained history, one copy.** `BirchTree`
  keeps every vector (needed for splits) as float64: 8·n·d bytes,
  plus the caller's array, plus CF sums. `fit` adopts its normalized
  input matrix as the history buffer instead of copying it again
  (an earlier version retained two copies). At 1M×384 the retained
  history is 3.07 GB — still the first wall at scale. A float32
  history would halve it but changes split numerics and is not
  done here.
- Per-wave O(G·d) terms (`centroids()`, tree copy, `dispersion()`)
  dominate at small wave sizes; wave=1 pays a ~29× sequential-
  semantics tax at 31k. Wave 1024 vs 256 at 31k changed speed only at
  noise level (4.86 s vs 5.07 s) but changed the partition. Split
  resolution no longer recomputes full-tree dispersion or rebuilds
  the assignment array per split (affected-group dispersion +
  contiguous labels with a member index; byte-identical results,
  measured effect small at 31k/100k — see
  `benchmarks/BOTTLENECKS.md`).
- Measured on this VM: 100k×384 blobs in 43.6 s. Laptop estimates
  (not measured): 1M×384 needs ≥ 16 GB RAM to be comfortable and
  extrapolates to roughly 1–2 h at wave 256. Dim 768 roughly doubles
  all O(·d) terms (10k: 1.47 s at d768 vs 0.81 s at d384, measured).

## Reproducing

See `benchmarks/REPRODUCING.md` for the full protocol and commands
(including the heavy 1M-row arms, which need a full laptop). In
short:

```sh
python -m unittest test_birch          # 53 tests
python benchmarks/bench_controlled.py  # controlled matrix (checkpointed)
python benchmarks/semantic_evidence.py # WordNet agreement + coherence
python benchmarks/profile_bottlenecks.py
```

The published 31,388-row WordNet matrix was selected by a private
workbench (corpus-driven synset selection), so the *identical rows*
cannot be regenerated from public sources. `REPRODUCING.md` gives a
public method-level regeneration procedure (Open English WordNet
2025 JSON + `all-MiniLM-L6-v2` embeddings) that reproduces the
method, not the exact rows; expect numbers in the same ballpark, not
identical. The synthetic tables reproduce exactly (fixed seed
20261125). Earlier single-run scripts that produced the pre-fix
numbers in `results/` are not included here (they depended on the
private selection process above); `bench_controlled.py` and
`semantic_evidence.py` are the supported harnesses.

## Origin

This library was developed inside a larger research system where
per-item BIRCH execution was the bottleneck. The clustering core was
extracted and is deliberately standalone: NumPy only, no dependency
on that system, usable without it.

## License

Apache License 2.0. See the [LICENSE](LICENSE) file.
