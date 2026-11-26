# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Parallel-scoring tests no longer assert bit-equality of scores across block shapes: some BLAS builds (e.g. scipy-openblas on Windows) round block matmuls ~1 ULP differently from the full matmul. Selection (indices) is still asserted exactly; scores are compared with tight tolerance. Host-measured Mode B results recorded (100k blobs: workers=1 45.9 s, workers=2 35.1 s, workers=4 31.5 s, identical groups; 1M pre-optimization baseline 3,769.6 s / 190,225 groups) — see `results/host-results-2026-11-26.json`.

### Changed

- Cluster features are now the numerically stable BETULA-style (`n`, `mu`, `S`) triple — count, mean vector, sum of squared deviations from the mean — maintained with Welford point updates and Chan group merges; dispersion is `S / n`. This replaces the classic (N, LS, SS) form (`SS/N − ‖LS‖²/N²`), which catastrophically cancels for data far from the origin. `CFTable` fields renamed: `ls` → `mu`, `ss` → `S` (only code touching the raw columns is affected). Verified equivalence-proven before adoption: byte-identical labels vs the previous implementation on blobs 1k / WordNet 31k / blobs 100k (132,388 items, zero flips), wall/RSS within noise, 53/53 tests; raw `CFTable` dispersion at offset 10⁸ now matches two-pass truth (was 0.0 vs true ~32).

### Added

- Parallel scoring blocks ("Mode B"): new `BirchConfig.workers` knob (default 1 = serial, no threads). With `workers > 1`, centroid rows are partitioned into contiguous blocks scored in parallel threads against the pinned snapshot; per-block top-k lists merge into the exact global top-k (score desc, group index asc), so clustering results are identical to the serial path (identical labels verified at 1k/31k/100k for workers 1/2/4; scores may differ by ~1 ULP across BLAS builds when block shapes change; 6 new tests, 53 total). The full wave×groups score matrix is never materialized in this mode. `BirchTree` reuses one lazily created thread pool (`close()` releases it); `place_wave` uses a per-call pool. Pin BLAS to one thread per worker to avoid oversubscription (see README).

### Performance

- Single retained data copy: removed a write-only per-wave vector list; `fit` adopts its normalized input matrix as the history buffer instead of copying it again. Peak RSS −23% at 31k/100k in local profiling; clustering results byte-identical.
- Split resolution computes dispersion for affected groups only (`CFTable.dispersion_rows`, bit-identical per-row values) instead of the full tree twice per split.
- Assignments are kept in one contiguous buffer with a group→member index; split resolution no longer rebuilds or rescans the full assignment array per split. Byte-identical results; 7 new regression tests (46 total).
- Per-wave split check computes dispersion for touched groups only (`dispersion_rows`) instead of the full tree after every wave. Byte-identical results; 1 new regression test (47 total).

## [0.1.0] - 2026-11-25

### Added

- Initial release of `batched-birch`: set-wise (batched) BIRCH clustering over embedding vectors with a NumPy-only core.
- Wave-based clustering core:
  - `place_wave` — places a whole wave of items against one pinned snapshot of the tree using matrix operations; returns assignments, new-group items, splits needed, and the updated tree.
  - `split_group` — resolves an overflowed group with the classic farthest-pair rule.
  - `BirchTree` — stateful wrapper with `fit` / `partial_fit` / `labels_`; resolves splits after each wave.
  - `BirchConfig` — clustering knobs (`min_similarity`, `max_dispersion`, `max_children`, `candidate_count`).
  - `CFTable` — tree state as typed columns (`n`, `ls`, `ss`, `parent`, `ids`) with vectorized `centroids()` and `dispersion()`.
  - `WaveResult` — result type returned by `place_wave`.
- Deterministic tie-breaking at every decision point (shortlist boundary, winner selection, split): fixed input order, no randomness, explicit tie-breaking to the lowest group index where applicable. Same input produces the same tree on any platform.
- First-wave consolidation: when the tree is empty, the first wave is consolidated into initial groups rather than treated as a sequence of singletons. The same within-wave consolidation applies to items that match no existing group in later waves.
- Input and configuration validation: shape/dtype checks on input vectors and validation of `BirchConfig` values at construction time. Non-finite values, zero rows, and dimension mismatches raise `ValueError`; inputs are unit-normalized on a copy (the caller's array is never mutated).
- `fit` reset semantics: calling `fit` on a `BirchTree` discards prior state and starts from an empty tree; `partial_fit` continues from existing state.

### Performance

- `BirchTree` history storage uses a capacity-doubling buffer instead of re-stacking the full vector history on every wave (was O(n²) copying). Byte-identical clustering results; ~38% faster at 31k items in the controlled benchmark.
