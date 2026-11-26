# Bottleneck review — batched-birch at 31k (and what breaks first)

Profile script: `benchmarks/profile_bottlenecks.py` (cProfile of
`BirchTree.fit`, WordNet 31,388×384, wave 256). Numbers below are
from this VM (AMD EPYC 9D25, 1.5 shared CPUs, 7.7 GB RAM). Absolute
seconds will differ on a laptop; the *shares* and asymptotics are
the point.

## Where the time goes (31k, wave 256, post-fix profile)

Total 11.5 s under cProfile at default threads (5.07 s median
unprofiled at threads=1 in the controlled bench — profiling and
thread oversubscription both inflate; use the bench JSON for wall
times, this file for structure).

| component | cum. s | calls | what it is |
|---|---|---|---|
| `_place_wave_impl` (self) | 2.62 | 123 | the `X @ C.T` matmul + candidate CF arithmetic |
| `_resolve_splits` | 2.38 | 123 | split loop incl. `split_group` (1.14) |
| `_shortlist` | 1.60 | 122 | argpartition (0.97) + tie handling |
| `einsum` (all) | 1.41 | 10,940 | dispersion / norms everywhere |
| `dispersion()` | 1.25 | 859 | full-tree dispersion, mostly to check 1–2 groups |
| `centroids()` | 1.20 | 122 | recomputed from scratch every wave |
| `_consolidate` | 0.97 | 123 | new-group formation (small — good) |
| `CFTable.copy` + `vstack` | 1.36 | 491+491 | tree copy per wave/split; `ls` regrowth |
| `_append_vectors` | 0.56 | 123 | history buffer appends (post-fix) |
| `_prepare_X` | 0.36 | 369 | re-normalizes split members every split |

## Complexity, honestly

Per wave with `w` items, `G` groups, dim `d`, shortlist `k=5`:

- Scoring is **O(w·G·d)** — one matmul, but `G` grows with `n`, so
  total fit cost is **O(n·G·d)** ≈ quadratic in the number of
  groups. At 31k (G≈4.8k) fine; at 100k blobs (G≈18.8k) ours took
  43.6 s median; extrapolating G∝n, 1M would be ~100× the 100k
  scoring work before memory even enters. There is no hierarchical
  routing in this version (flat tree by design) — a CF-tree with
  internal nodes is the known fix and is explicitly out of scope.
- Candidate arithmetic materializes a `(w, k, d)` array per wave
  (256×5×384×8 B ≈ 3.9 MB) — allocation churn, not a wall.
- `centroids()` and `tree.copy()` are **O(G·d) per wave**, and
  `dispersion()` is O(G·d) per call (859 calls: every wave + twice
  per split, to check single groups). [2026-11-25: both the
  twice-per-split calls and the per-wave call in `_place_wave_impl`
  now compute dispersion for the affected/touched groups only
  (`dispersion_rows`) — see addendum.] At wave=1 these per-wave
  O(G·d) terms run per *item*: wave=1 on 31k took 147.8 s (vs
  5.07 s at wave 256) — a 29× sequential-semantics tax, dominated
  by per-wave fixed costs, not the 1×G matmul.
- Splits: 368 at 31k. Each split re-normalizes members
  (`_prepare_X`), builds a members×members Gram matrix
  (members ≤ 16 + wave overflow, so small), copies the whole tree,
  and re-checks dispersion over all G groups. Split *count* is
  modest; split *bookkeeping* is O(G·d) each.
- `labels_` concatenates the assignment list on every access;
  `_resolve_splits` accesses it per split. O(n) per split —
  negligible at 31k (368 × 31k ints), 368M+ element copies at 1M
  scale. Cache it if splits ever get hot at scale.
  [Fixed 2026-11-25: contiguous label buffer + group→member index;
  split resolution no longer rebuilds or rescans assignments —
  see addendum.]

## Memory growth

`BirchTree` retains **every vector** (needed to gather split
members): O(n·d) float64 = 8·n·d bytes, plus the caller's array,
plus CF `ls` (G·d float64). At 31k×384: history 96 MB. At
100k×384: 307 MB. At **1M×384: history alone is 3.07 GB float64**,
plus input copy in `_prepare_X` (another 3.07 GB transient), plus
`ls` (~150k groups × 384 × 8 B ≈ 460 MB at blobs-like G/n) —
**>6.5 GB peak, over this VM's cap before scoring starts.**
[2026-11-25 retention fix: `fit` adopts the `_prepare_X` copy as
the history buffer, so the transient copy and the history are now
the same matrix — one fewer n·d copy at peak than the arithmetic
above; the 1M rerun will re-measure. See addendum.] Both
1M arms recorded `memory_error` under a 5.5 GB guard (34 s / 52 s
to fail, mostly generation). 1M is a host arm, pending host run —
not a failure of the algorithm, a capacity
fact: float64 history retention is the binding constraint. A
float32 history (or spilling members) would halve it; not done
here (would change split numerics — needs its own verification).

## Perf fix applied (bookkeeping only, labels identical)

**Before:** `BirchTree._partial_fit_prepared` did
`np.vstack([self._all_X, Xn])` every wave — O(history) copy per
wave, O(n²·d) total. Pre-fix profile: `vstack` was the #1 cost
(613 calls, 2.04 s of 6.95 s profiled). At wave=1 it is
catastrophic (31k full-history copies).

**Fix:** capacity-doubling buffer (`_append_vectors`); `_all_X`
is now a property returning a view. Amortized O(n) appends. No
math changed — same values, same order.

**Evidence:** all 39 unit tests pass; labels on WordNet 10k
prefix and full 31k (wave 256) are **byte-identical** before/after
(SHA-256 `194edd4d…` / `a9012023…` both sides); 31k wall at
threads=1 went 7.85 s → 4.86 s (−38%) in the before/after check;
wave=1 at 31k completes in 147.8 s (pre-fix it would not finish
in the 300 s cap). Post-fix `vstack` remains only in
`append_groups`/`split_group` `ls` regrowth (491 calls, 0.68 s) —
same quadratic pattern at group scale, documented, not fixed
(preallocating `CFTable` touches copy semantics everywhere).

## Scaling limits (this VM vs a typical laptop)

- **VM (measured):** 100k×384 blobs: ours 43.6 s median, peak RSS
  in frags; sklearn-fit 211 s. 1M: infeasible (memory math above).
- **Laptop (estimate, not measured):** 8–16 GB RAM fits 1M×384
  history + input (~6.5 GB peak) only just; time extrapolates to
  roughly 1–2 h for ours at wave 256 (O(n·G·d) with G≈150–190k)
  — host arm, pending host run. Dim 768 doubles all O(·d) terms
  (10k d768 ours 1.47 s vs d384 0.81 s — measured).
- First wall at scale is **memory (float64 history)**, second is
  **flat O(w·G·d) scoring**. Neither is fixed by tuning wave size;
  wave 1024 vs 256 at 31k: 4.86 s vs 5.07 s (noise-level), groups
  4,965 vs 4,796 (partition changes, speed doesn't).

## Addendum 2026-11-25 — pre-publication optimizations (Sol feedback 1/3/4)

Three bookkeeping fixes, each profiled independently with
byte-identical labels vs the pre-change baseline at WordNet 31k
and blobs 100k (threads=1, single runs; this VM ran ~7–9× slower
than during the controlled bench above, so read the *deltas*,
not the absolutes; equivalence baselines in
`results/equiv-baseline-*.npy`):

1. **Single retained copy.** Removed the write-only per-wave
   vector list; `fit` adopts its normalized input matrix as the
   history buffer (waves are views; appends just advance a
   length). Peak RSS: 31k 400→309 MB, 100k 1193→922 MB (−23%
   both ≈ one n·d float64 copy). Wall unchanged.
2. **Affected-group dispersion in split resolution.**
   `CFTable.dispersion_rows(rows)` (bit-identical per-row values);
   `_resolve_splits` no longer computes full-tree dispersion
   twice per split. Wall: 100k 386.3→384.4 s (splits are
   infrequent at these scales; the saving grows with G×splits).
   [Follow-up done 2026-11-25: the per-wave full-tree
   `dispersion()` in `_place_wave_impl` (only touched groups were
   read) now also uses `dispersion_rows(touched)` — see the
   2026-11-25 per-wave entry below.]
3. **Contiguous labels + member index.** Assignments live in one
   growable int64 buffer; a group→member-positions index (arrival
   order) replaces per-split `labels_` concatenation, full scans,
   and chunk remaps. Wall: 100k 384.4→380.6 s; 31k 37.8→37.3 s.

Final state vs baseline: labels byte-identical at both scales
(4,796 / 18,824 groups); tracemalloc peak at 31k 270.9→193.6 MB;
46 unit tests pass (39 original + 7 regression). The 1M host
rerun after these fixes is **pending** (the in-flight host run is
the pre-fix baseline). Remaining known costs, unchanged:
O(w·G·d) flat scoring, per-wave `centroids()`/tree-copy,
`ls` vstack regrowth in append/split, float64
history retention. Hierarchical/indexed candidate routing stays
a separate architectural milestone.

## Addendum 2026-11-25 — per-wave dispersion micro-fix

The deferred follow-up from item 2 above: `_place_wave_impl`
computed full-tree `dispersion()` after every wave's merge but
read only the touched groups (`np.unique(assignment)`). It now
calls `dispersion_rows(touched)` — per-wave dispersion cost drops
from O(G·d) to O(touched·d), touched ≤ wave size (256). At 1M
scale (G≈190k, ~3,900 waves) that is the difference between
~190k·384 and ≤256·384 arithmetic per wave. Bit-identical
per-row values; labels byte-identical vs the equivalence
baselines at WordNet 31k and blobs 100k (threads=1, single runs
back-to-back on this VM; absolutes vary with shared-CPU load,
deltas within the session are what count):

| dataset | before | after | peak RSS before→after |
|---|---|---|---|
| WordNet 31k | 46.9 s | 37.5 s | 265.7 → 265.8 MB |
| blobs 100k | 383.1 s | 381.5 s | 822.4 → 833.3 MB |

The 100k wall delta is noise-level (scoring dominates there);
the 31k delta (−20%) is larger than the profiled dispersion
share alone would predict, so treat it as an upper bound
contaminated by VM variance. The structural saving is the
claim; the 1M host rerun is where it should show. Regression
test: `test_place_wave_split_check_reads_only_touched_rows`
(full-tree `dispersion()` patched to raise; result identical,
rows requested == touched set). 47 unit tests pass.

## Addendum 2026-11-26 — Mode B parallel scoring blocks (Sol item 2)

Sol's item 2 (blockwise centroid scoring + exact top-k without
materializing the wave×groups matrix) is implemented, combined
with thread parallelism ("Mode B"): `BirchConfig.workers`
(default 1 = exactly the serial path, no threads). With
workers > 1, centroid rows are partitioned into contiguous
blocks (over groups only, never the feature dimension); each
worker thread scores its block against the pinned snapshot and
returns a local top-k; the merge is the exact global top-k
(score desc, group index asc). `BirchTree` reuses one lazily
created `ThreadPoolExecutor` (`close()` releases it); the free
`place_wave` uses a per-call pool. BLAS must be pinned to 1
thread per worker (env, before NumPy import) — W workers × T
BLAS threads oversubscribes. The Python-level residue (merge,
acceptability, consolidation, splits) stays serial, so speedup
is bounded by the scoring share; asymptotics are unchanged
(still O(n·G·d) total scoring work). Hierarchical/indexed
candidate routing remains the separate v0.2 milestone.

Equivalence (hard gate, threads=1): labels byte-identical vs
the pre-change baselines (`results/equiv-baseline-*.npy`; the
blobs-1k baseline was captured pre-edit for this change) at
blobs 1k (103 groups), WordNet 31k (4,796), blobs 100k
(18,824) for workers ∈ {1, 2, 4} — all 9 combinations.

VM benchmark (subprocess per arm, warmup + fresh model per rep,
BLAS pinned to 1; **this VM has 1.5 shared CPUs — these numbers
measure correctness + threading overhead, NOT laptop speedup**;
the `__workers2`/`__workers4` arms in `bench_controlled.py`
are the laptop measurement):

| dataset | workers | wall (s) | peak RSS (MB) |
|---|---|---|---|
| blobs 1k (median of 3) | 1 | 0.055 | 53.7 |
| blobs 1k | 2 | 0.051 | 54.6 |
| blobs 1k | 4 | 0.055 | 54.1 |
| blobs 100k (1 rep) | 1 | 381.4 | 961.8 |
| blobs 100k | 2 | 269.6 | 1019.4 |
| blobs 100k | 4 | 279.1 | 1005.7 |

Reading: at 1k threading is invisible (55 ms total fit). At
100k, workers=2/4 run at ~0.71–0.73× serial — two threads can
use the ~1.5 shared CPUs for the block matmuls; 4 adds nothing
beyond 2 here (oversubscription begins; on an 8-core laptop the
knee should sit higher). Peak RSS +5–6% at 100k with
workers > 1 (concurrent per-block score matrices + thread
stacks; the avoided full score matrix is only ~38 MB at this
scale — at 1M it is ~390 MB and the trade flips). The
equivalence-gate runs (single fits, same process) showed the
same ratios at 31k (38.3 s serial vs 27.4/27.6 s at
workers 2/4). 53 unit tests pass (47 + 6 new: validation,
blockwise-vs-serial shortlist equality incl. a crafted
cross-block tie, fit/place_wave equivalence, pool lifecycle).

## Addendum 2026-11-26 — stable (n, mu, S) cluster features adopted

The cluster feature switched from the classic (N, LS, SS) form to
the numerically stable BETULA-style (n, mu, S) triple (Welford
point updates, Chan group merges; dispersion = S/n). `CFTable`
columns renamed: `ls` → `mu`, `ss` → `S` — the `ls` mentions in the
dated profile entries above refer to the same G·d float64 column,
now holding group means instead of vector sums.

Why: the classic dispersion `SS/N − ‖LS‖²/N²` catastrophically
cancels for data far from the origin. The public API path was
already safe (inputs are unit-normalized), but raw `CFTable` use
was not: at offset 10⁸ the classic form returned dispersion 0.0
for data whose true dispersion is ~32. The stable form matches
two-pass truth to ~10⁻¹⁵ relative at every offset tested
(0 … 10⁸).

Cost: none measured. Equivalence-proven before adoption —
byte-identical labels vs the (N, LS, SS) implementation on blobs
1k, WordNet 31,388, and blobs 100k (132,388 items, zero flips;
final-group dispersion differences ≤ ~10⁻¹⁵, while the closest
group sits ~10⁻⁵ from the max_dispersion threshold). Wall/RSS
within noise (+0.8% / +0.02% wall at 31k/100k in the comparison
run). 53/53 tests pass. (The stable-CF variant and its comparison
harness were kept with the development tree; the key numbers are
recorded above.)

## Addendum 2026-11-26 — Mode B on real hardware (laptop)

Host: AMD Ryzen 7 7735HS, 8 cores, Windows, Python 3.14,
scipy-openblas 0.3.34. Synthetic blobs 100k, wave 256, BLAS
pinned to 1 thread per worker:

| arm | wall (s) | groups | vs workers=1 |
|---|---|---|---|
| workers=1 | 45.924 | 18,824 | 1.00× |
| workers=2 | 35.078 | 18,824 | 1.31× |
| workers=4 | 31.5 | 18,824 | 1.46× |
| workers=1, BLAS unpinned | 45.115 | 18,824 | 1.02× |

Reads: (1) Multithreaded BLAS alone buys nothing at this scale —
Task Manager showed ~2 of 16 logical processors busy; the
bottleneck is the single-threaded residue, not the matmul.
(2) Mode B scaling flattens (w2→w4: 1.11×); fitting Amdahl gives
a parallel (scoring) fraction of ~45%, capping total gain near
1.8× regardless of worker count. The next lever is the serial
residue (consolidation, merge, splits), not more scoring threads.
(3) Selection is exact at every worker count on this BLAS too
(18,824 groups; frag label hashes match baselines). Scores can
differ by ~1 ULP across BLAS builds when block shapes change —
tests compare scores with tolerance, indices exactly.
(4) The 1M arm (pre-optimization code, workers=1, pinned):
3,769.6 s, 190,225 groups, ~8 GB peak (three retained copies).
Post-fix peak should be ~3 GB lower; rerun on current code
pending. Full host record: `results/host-results-2026-11-26.json`.
