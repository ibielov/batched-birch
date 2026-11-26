# Semantic evidence — corrected (new code, WordNet 31k)

Raw data: `results/semantic-evidence-2026-11-25.json` (script:
`benchmarks/semantic_evidence.py`). Labels:
`benchmarks/wordnet-birch-labels-new.npy`. The pre-fix numbers in
`results/wordnet-grouping-comparison-2026-11-25.json` and
`results/three-way-*.json` are kept untouched as the before picture.

## What changed in the code (why numbers moved)

Three P0 correctness fixes landed before this run: (1) within-wave
consolidation — unassigned items (including the whole first wave)
now form groups among themselves instead of becoming singletons;
(2) inputs are unit-normalized on a copy; (3) `fit()` resets.
Consequence at WordNet 31k, wave 256, default config:

| | before (single run) | after (this run) |
|---|---|---|
| groups | 5,209 | **4,796** |
| mean size | 6.03 | 6.55 |
| singleton groups | 1,133 (21.8% of groups) | 881 (18.4% of groups; 2.8% of items) |
| coherence (pair-weighted) | 0.506 | 0.504 |
| member→centroid | 0.755 | 0.760 |
| NMI vs depth-3 ancestors | 0.485 | 0.485 |

At the 1,000-row prefix the change is larger: 591 → 353 groups
(singleton items 72.8% → 14.0%). The old first-wave behavior
manufactured singletons; the old 1k "quality" numbers are not
comparable and should not be quoted for the current code.

## Agreement with WordNet's own groupings (new code)

| reference | ref groups | purity (ours→ref) | purity (ref→ours) | NMI | ARI |
|---|---|---|---|---|---|
| lexnames (41 files) | 41 | 0.527 | 0.014 | 0.372 | 0.003 |
| hypernym ancestors, depth 2 | 1,375 | 0.536 | 0.059 | 0.372 | 0.001 |
| hypernym ancestors, depth 3 | 3,155 | 0.488 | 0.126 | 0.485 | 0.002 |

Reading: our 4,796 clusters sit at roughly depth-3 granularity.
ARI near zero is expected at this granularity mismatch (ARI
penalizes the 41-vs-4,796 split heavily); NMI is the informative
column. Agreement barely moved vs the old code (depth-3 NMI
0.485 → 0.485) — consolidation changed group *shape*, not
WordNet alignment.

**Why WordNet agreement is informative but not definitive.**
Lexicographer files are a lexicographic filing system (by syntactic
and broad semantic class), not a topical ground truth: a sailing
cluster spanning artifact/motion/act files disagrees with WordNet
and can still be topically right. Conversely, agreeing with
WordNet only shows the embeddings recover lexicographic structure,
not that clusters are semantically correct. And **coherence is not
correctness**: coherence measures embedding geometry (mean pairwise
cosine). A cluster can be geometrically tight and semantically
wrong — see the "plane" cluster below (coherence is unremarkable,
the grouping is a word-form accident). No cluster fell below 0.20
coherence (p10 = 0.449), so mechanical coherence thresholds do not
detect the known failure mode; only reading does.

## The "more reasonable than WordNet" claim — corrected

The old README/audit classified 3,112 clusters as "more reasonable
than WordNet" vs 465 the other way. **That classification is
HEURISTIC**: a mechanical rule (cross-lexname clusters whose
embedding coherence is high and whose cross-lexname cosine is close
to within-lexname cosine) plus the author's hand-read of a small
sample. It is not independent review, not inter-annotator measured,
and the rule's thresholds (coherence ≥ 0.30, gap < 0.15) are
choices, not findings.

Re-derived here with the rule stated explicitly in
`semantic_evidence.py` (agreement = single lexname;
cross_lexname_coherent = coherence ≥ 0.30 and within−cross gap
< 0.15; split_candidate = coherence < 0.20 or gap ≥ 0.15; else
borderline), on the new code's 3,915 multi-member clusters:

- cross_lexname_coherent: **2,915 clusters** (24,347 members) —
  the heuristic's "ours more reasonable" bucket.
- split_candidate: **475 clusters** (3,845 members).
- agreement: 525 clusters (2,315 members). Borderline: 0.

These counts describe the rule, not semantic truth. Quote them
only with the HEURISTIC label attached.

## Representative false merges (shared-headword binding)

Mechanical screen (size ≥ 4, one headword ≥ 50% of members, ≥ 3
lexnames) flags 720 clusters — many are legitimate polysemy sets
(all senses of "hold", "place"). The failure mode is narrower:
clusters united by word *form* across unrelated meanings.
Re-derived from new-code labels:

- **"plane" (cluster 3350, 11 members)** — London plane (tree),
  carpenter's plane / match plane / router plane (woodworking),
  plane seat (airliner), plane table (surveying), Cartesian plane
  + tangent plane (math), "plane" as level of existence. This is
  the old audit's plane mashup, still present in the new code.
- **"time" (largest time cluster 353, 16 members)** — mostly
  time-period senses (work time, travel time, lead time, access
  time, geological/cosmic time). The old audit's worst time cluster
  (timing + time zone + time deposit) does not recur in that form;
  the new grouping is defensible. "Time deposit" senses now sit
  with finance vocabulary. Reported as-is: one known failure mode
  persists (plane), one improved (time).
- "rock" (cluster 1603) mixes rock music, rock sugar, and rocking
  motion — form-bound, geometrically coherent (0.485).

Bottom line: embedding coherence cannot distinguish these from
good topical clusters. Any downstream use that needs semantic
correctness needs a reading/audit step, not a coherence threshold.

## Provenance note

The 31,388 rows were selected by a private workbench (corpus-driven
synset selection); the clustering and all metrics here were computed
from the public Open English WordNet 2025 JSON + cached vectors.
See `REPRODUCING.md` for the public method-level regeneration
procedure (it reproduces the method, not the identical rows).
