# Contributing

Thanks for your interest in `batched-birch`. This is a small, focused library — contributions that keep it that way are welcome.

## Development setup

```sh
git clone <repo-url>
cd batched-birch
pip install -e .[dev]
```

The `dev` extra pulls in `scikit-learn`, which is only needed for the benchmarks. The library core itself depends on NumPy alone.

## Running tests

```sh
python -m unittest test_birch -v
```

Tests live at the repository root in `test_birch.py` and import the installed `batched_birch` package.

## Code style

- **NumPy-only core.** The `batched_birch` package must not import anything beyond the Python standard library and NumPy. Embedders, stores, and other integrations belong in adapters outside the core.
- **Determinism required.** Same input must produce the same tree on any platform. Use explicit tie-breaking at every decision point; do not rely on hash order, dict/set iteration order for algorithmic choices, or platform-specific float behavior.
- **No randomness without a seed.** If a change needs randomness (it probably doesn't), it must accept an explicit seed parameter and be fully reproducible from that seed.
- Keep the core small. Prefer clear vectorized NumPy over clever tricks; add a comment where the vectorization is non-obvious.

## Running benchmarks

Benchmark scripts live in `benchmarks/`. Start with `benchmarks/REPRODUCING.md` — it has the exact commands to regenerate every published table. The current controlled suite:

- `bench_controlled.py` — the controlled head-to-head vs scikit-learn's `Birch` (subprocess arms, imports outside timing, warmups, medians + IQR, thread control, environment recorded per arm; checkpointed fragments under `results/frags-2026-11-25/`). Needs the `dev` extra (`pip install -e .[dev]`).
- `gen_synthetic.py` — standalone seeded synthetic-data generator (uniform sphere, Gaussian blobs, duplicate-heavy). NumPy only, no other dependencies.
- `semantic_evidence.py` — WordNet grouping-agreement metrics (NMI/ARI/coherence) for the current code.
- `profile_bottlenecks.py` — profiling harness behind `benchmarks/BOTTLENECKS.md`.
- Heavy arms (1M scale) exceed small-VM capacity: run `python benchmarks/bench_controlled.py --only syn-blobs-1000000 --timeout 14400` on a full laptop (≥16 GB RAM recommended).

Earlier single-run scripts that produced the pre-fix numbers in `results/` are not included (they depended on an external process for the exact WordNet selection; see `REPRODUCING.md`). Prefer the controlled suite for new measurements.

## Pull requests

- Include tests for any behavior change. New features and bug fixes should both come with a test case in `test_birch.py` (or a clear explanation of why a test isn't feasible).
- Run the full test suite before submitting: `python -m unittest test_birch -v`.
- Keep changes focused — one logical change per PR.
- Update `CHANGELOG.md` under `[Unreleased]` for user-visible changes.
