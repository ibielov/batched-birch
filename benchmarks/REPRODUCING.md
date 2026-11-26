# Reproducing the batched-birch benchmarks

Date of published run: 2026-11-25. All numbers below come from
`benchmarks/bench_controlled.py` writing
`results/benchmark-2026-11-25.json` (per-arm fragments in
`results/frags-2026-11-25/`), plus the semantic-evidence run in
`results/semantic-evidence-2026-11-25.json`. The older single-run
files in `results/` (`birch-vs-sklearn-2026-11-25.json`,
`wordnet-grouping-comparison-2026-11-25.json`, `three-way-*.json`)
are the *before* picture for the pre-fix code and are kept
untouched; do not expect them to match — see "What changed" in
`SEMANTIC-EVIDENCE.md` and `BOTTLENECKS.md`.

## Environment

- Python >= 3.10, NumPy >= 1.24 (core). Benchmarks additionally
  need scikit-learn (`pip install scikit-learn`, or
  `pip install -e .[dev]` from the package root).
- Published run: Python 3.12.3, NumPy 2.5.5, scikit-learn 1.9.1,
  AMD EPYC 9D25 (4 vCPUs visible, 1.5 shared CPUs), 7.7 GB RAM,
  Linux. Every result fragment records its own environment; check
  the `environment` field before comparing across machines.
- BLAS threads are pinned to 1 for comparability
  (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`,
  set by the driver per subprocess). Two arms
  (`*__threads-default`) run at default threads to show sensitivity.
  On the published VM, default threads were *slower* for
  batched-birch (oversubscription on 1.5 shared CPUs); on a real
  multi-core laptop the sign may flip. Never compare a threads=1
  number against a default-threads number.

## Protocol (what "controlled" means here)

1. One subprocess per arm. Imports and data loading are outside the
   timed region; only the fit call is timed. sklearn labels for
   metrics come from `predict(X)` (nearest subcluster center),
   timed separately as `predict_s_last_rep` and never inside
   `wall_s` — at 100k rows predict alone is a 100k×groups×dim
   matmul that would otherwise swamp the fit comparison. (An
   earlier single-run script timed the sklearn import too, which
   inflated small-scale sklearn times by ~2 s — one reason the
   before/after table in the report looks different at 1k.)
2. One untimed warmup fit on a ≤2,000-row prefix, then timed reps
   with a fresh model each rep: 3 reps up to 31k rows, 2 reps at
   100k, 1 rep for wave_size=1 diagnostics and the 1M attempt.
   Reported wall time is the median; spread is the IQR (range for
   2 reps). Label hashes verify reps are identical.
3. Per-arm timeout 900 s (300 s for wave_size=1 on WordNet 31k).
   Timeouts and MemoryErrors are recorded as results.
4. Arms: `ours` = `BirchTree(BirchConfig()).fit(X, wave_size=W)`
   (cosine gate 0.5, dispersion ≤ 0.5, capacity 16);
   `sklearn-fit` / `sklearn-partial` =
   `Birch(threshold=0.5, branching_factor=50, n_clusters=None)`
   (threshold is a Euclidean subcluster *radius*, not a cosine
   gate; partial_fit in waves of 256); `sklearn-matched` =
   sklearn fit at threshold 0.865, tuned so its WordNet-31k group
   count approximates ours. Matched granularity is not matched
   semantics: cosine gate + dispersion + capacity vs radius +
   branching factor produce different partitions at the same group
   count. No arm here supports a claim of universal superiority —
   only speed/quality trade-offs on these distributions.

## Regenerating every published table

From the package root (`public/birch/`):

```sh
# 0. sanity: 46 unit tests
python -m unittest test_birch

# 1. synthetic data (optional; bench_controlled generates in-process
#    with the same seeds — these commands materialize the same arrays)
python benchmarks/gen_synthetic.py --dist blobs --n 10000 --dim 384 \
    --seed 20261125 --n-blobs 50 --spread 0.8 --out /tmp/syn-blobs-10k
python benchmarks/gen_synthetic.py --dist uniform --n 10000 --dim 384 \
    --seed 20261125 --out /tmp/syn-uniform-10k
python benchmarks/gen_synthetic.py --dist duplicates --n 10000 --dim 384 \
    --seed 20261125 --dup-frac 0.5 --out /tmp/syn-dup-10k

# 2. full controlled matrix (checkpointed; rerun to resume)
python benchmarks/bench_controlled.py
#    subsets:  --only wordnet-31388 | --only syn-blobs-100000 | --list

# 3. semantic evidence on WordNet 31k (needs the vectors + texts and
#    the Open English WordNet 2025 JSON zip; see below)
python benchmarks/semantic_evidence.py

# 4. bottleneck profile at 31k
python benchmarks/profile_bottlenecks.py
```

Fixed seeds: synthetic seed `20261125` everywhere; blob parameters
`n_blobs=50, spread=0.8`; duplicate pool fraction `0.5`. WordNet
rows are a fixed prefix order (1k = first 1,000 rows of the 31,388).

## Public WordNet / MiniLM regeneration procedure (method, not rows)

The published 31,388-row matrix was selected by a private workbench
(synsets whose lemmas intersect a specific corpus vocabulary,
closed over hypernym ancestors, ranked by corpus document
frequency). That corpus is not public, so the *identical rows*
cannot be regenerated from public sources. What can be reproduced
publicly is the *method*:

1. Download **Open English WordNet 2025** (JSON) from its public
   source: https://github.com/globalwordnet/english-wordnet
   (release `2025`, asset `english-wordnet-2025-json.zip`).
2. Load `noun.*.json` + `verb.*.json`; each synset has `members`
   (lemmas), `definition`, and `hypernym` ids.
3. Render one sentence-form text per synset, e.g.
   `"A <lemma> is a <hypernym lemma>. The <lemma> is <definition>."`
   (plus an "also called …" clause for extra lemmas). The exact
   template used for the published rows lives in the private
   workbench; any fixed template reproduces the method.
4. Embed with **sentence-transformers `all-MiniLM-L6-v2`**
   (384-dim, unit-normalized):
   `pip install sentence-transformers`, then
   `SentenceTransformer('all-MiniLM-L6-v2').encode(texts, normalize_embeddings=True)`.
5. Save `wordnet-vectors.npy` (float32) + `wordnet-texts.json`
   (`{"ids": [...], "texts": [...]}`) into `benchmarks/`, then run
   steps 2–3 above (`BIRCH_WORDNET_VECTORS` / `BIRCH_WORDNET_TEXTS`
   override locations).

Expect group counts and NMI/ARI in the same ballpark, not identical
numbers: row selection, template wording, and embedder build all
shift the geometry. The synthetic tables (step 1–2) reproduce
exactly on any machine with the same NumPy major version.

## Files

| file | contents |
|---|---|
| `results/benchmark-2026-11-25.json` | merged controlled results (all arms) |
| `results/frags-2026-11-25/*.json` | per-arm fragments (checkpoint unit) |
| `results/semantic-evidence-2026-11-25.json` | WordNet agreement + coherence, new code |
| `benchmarks/SEMANTIC-EVIDENCE.md` | reading of the semantic evidence, corrections |
| `benchmarks/BOTTLENECKS.md` | profile, complexity, scaling limits |
