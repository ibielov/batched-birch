"""Semantic evidence for the NEW batched-birch code on WordNet 31k.

Fits BirchTree (defaults, wave 256) on the cached WordNet vectors,
then reports — together, in one JSON — group shape, embedding
coherence, and agreement with WordNet's own groupings (lexicographer
files; hypernym ancestors at depth 2 and 3). Also emits a HEURISTIC
cross-lexname classification and representative shared-headword
false merges. See SEMANTIC-EVIDENCE.md for how to read this; in
short: WordNet agreement is informative, not definitive, and
embedding coherence is not semantic correctness.

Inputs (env overrides):
  BIRCH_WORDNET_VECTORS  default benchmarks/wordnet-vectors.npy
  BIRCH_WORDNET_TEXTS    default benchmarks/wordnet-texts.json
  BIRCH_WORDNET_ZIP      default benchmarks/english-wordnet-2025-json.zip
                         (Open English WordNet 2025, public source:
                         github.com/globalwordnet/english-wordnet)
Output: results/semantic-evidence-2026-11-25.json
        + benchmarks/wordnet-birch-labels-new.npy

Note on provenance: the 31,388-row selection was corpus-driven via a
private workbench (see REPRODUCING.md). This script measures OUR
clustering of those rows; it does not need the workbench at runtime.
Requires: numpy, scikit-learn (metrics only).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path[:0] = [str(PKG)]

VECTORS = Path(os.environ.get('BIRCH_WORDNET_VECTORS', str(HERE / 'wordnet-vectors.npy')))
TEXTS = Path(os.environ.get('BIRCH_WORDNET_TEXTS', str(HERE / 'wordnet-texts.json')))
ZIP = Path(os.environ.get('BIRCH_WORDNET_ZIP', str(HERE / 'english-wordnet-2025-json.zip')))
OUT_JSON = PKG / 'results' / 'semantic-evidence-2026-11-25.json'
OUT_LABELS = HERE / 'wordnet-birch-labels-new.npy'


def load_with_lexnames(wordnet_dir):
    synsets, lexname = {}, {}
    for path in sorted(Path(wordnet_dir).glob('noun.*.json')) + \
            sorted(Path(wordnet_dir).glob('verb.*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        synsets.update(data)
        for sid in data:
            lexname[sid] = path.name[:-len('.json')]
    return synsets, lexname


def compute_depths(synsets):
    depth, visiting = {}, set()

    def depth_of(sid):
        if sid in depth:
            return depth[sid]
        if sid in visiting:
            return 0
        visiting.add(sid)
        hypers = [h for h in (synsets.get(sid) or {}).get('hypernym') or []
                  if h in synsets]
        d = 0 if not hypers else 1 + max(depth_of(h) for h in hypers)
        visiting.discard(sid)
        depth[sid] = d
        return d

    for sid in synsets:
        depth_of(sid)
    return depth


def ancestor_at_depth(synsets, depth, sid, target):
    cur = sid
    while depth[cur] > target:
        hypers = [h for h in (synsets.get(cur) or {}).get('hypernym') or []
                  if h in synsets]
        if not hypers:
            break
        cur = min(hypers, key=lambda h: (-depth[h], h))
    return cur


def purity(true_labels, cluster_labels):
    by_cluster = defaultdict(Counter)
    for t, c in zip(true_labels, cluster_labels):
        by_cluster[c][t] += 1
    return sum(c.most_common(1)[0][1] for c in by_cluster.values()) / len(true_labels)


def compare(name, true_labels, cluster_labels, report):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    entry = {
        'original_groups': len(set(true_labels)),
        'our_groups': len(set(int(c) for c in cluster_labels)),
        'purity_ours_vs_original': round(purity(list(true_labels), list(cluster_labels)), 4),
        'purity_original_vs_ours': round(purity(list(cluster_labels), list(true_labels)), 4),
        'nmi': round(float(normalized_mutual_info_score(
            true_labels, cluster_labels, average_method='arithmetic')), 4),
        'ari': round(float(adjusted_rand_score(true_labels, cluster_labels)), 4),
    }
    report['comparisons'][name] = entry
    print(f"{name}: {entry}", flush=True)


def pair_cos(X, idx_a, idx_b):
    """Mean cosine over cross pairs (a in A, b in B), unit rows."""
    if not len(idx_a) or not len(idx_b):
        return None
    return float((X[idx_a] @ X[idx_b].T).mean())


def within_cos(X, idx):
    if len(idx) < 2:
        return None
    G = X[idx] @ X[idx].T
    iu = np.triu_indices(len(idx), k=1)
    return float(G[iu].mean())


HEADWORD_RE = re.compile(r'^(?:A|An|To)\s+(\S+)')


def main():
    from batched_birch import BirchConfig, BirchTree

    vectors = np.load(VECTORS)
    texts_doc = json.loads(TEXTS.read_text(encoding='utf-8'))
    ids, texts = texts_doc['ids'], texts_doc['texts']
    assert len(ids) == len(vectors), (len(ids), vectors.shape)
    X = np.ascontiguousarray(vectors, dtype=np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    print(f'vectors {X.shape}', flush=True)

    report = {'comparisons': {}, 'provenance':
              'rows selected by private workbench; clustering + metrics '
              'computed here from public WordNet zip + cached vectors'}

    with TemporaryDirectory() as tmp:
        with zipfile.ZipFile(ZIP) as archive:
            archive.extractall(tmp)
        # zip may nest a top folder
        root = Path(tmp)
        if not list(root.glob('noun.*.json')):
            for sub in root.iterdir():
                if sub.is_dir() and list(sub.glob('noun.*.json')):
                    root = sub
                    break
        synsets, lexname = load_with_lexnames(root)
    print(f'synsets {len(synsets)}', flush=True)
    assert all(sid in synsets for sid in ids)

    lex_labels = [lexname[sid] for sid in ids]
    depth = compute_depths(synsets)
    anc2 = [ancestor_at_depth(synsets, depth, sid, 2) for sid in ids]
    anc3 = [ancestor_at_depth(synsets, depth, sid, 3) for sid in ids]

    cfg = BirchConfig()
    t0 = time.perf_counter()
    tree = BirchTree(cfg).fit(vectors, wave_size=256)
    wall = time.perf_counter() - t0
    labels = tree.labels_
    np.save(OUT_LABELS, labels)
    counts = np.bincount(labels)
    print(f'BIRCH: {tree.n_groups} groups in {wall:.1f}s', flush=True)

    # Coherence: pair-weighted + per-cluster distribution + member->centroid.
    sums = np.zeros((tree.n_groups, X.shape[1]))
    np.add.at(sums, labels, X)
    sq = np.einsum('ij,ij->i', sums, sums)
    pair_w = float((sq - counts).sum() / (counts * (counts - 1)).sum())
    per_cluster = []
    for g in np.where(counts >= 2)[0]:
        c = within_cos(X, np.where(labels == g)[0])
        if c is not None:
            per_cluster.append(c)
    per_cluster = np.array(per_cluster)
    cent = sums / counts[:, None]
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-300)
    m2c = float(np.einsum('ij,ij->i', X, cent[labels]).mean())

    report['birch'] = {
        'config': {'min_similarity': 0.5, 'max_dispersion': 0.5,
                   'max_children': 16, 'candidate_count': 5, 'wave_size': 256},
        'groups': int(tree.n_groups), 'wall_seconds': round(wall, 2),
        'size_mean': round(float(counts.mean()), 3),
        'size_median': float(np.median(counts)),
        'size_p90': float(np.percentile(counts, 90)),
        'size_max': int(counts.max()),
        'singleton_groups': int((counts == 1).sum()),
        'singleton_group_rate': round(float((counts == 1).mean()), 4),
        'singleton_item_rate': round(float((counts == 1).sum() / len(labels)), 4),
        'coherence_pair_weighted': round(pair_w, 4),
        'coherence_per_cluster_median': round(float(np.median(per_cluster)), 4),
        'coherence_per_cluster_p10': round(float(np.percentile(per_cluster, 10)), 4),
        'coherence_per_cluster_p90': round(float(np.percentile(per_cluster, 90)), 4),
        'frac_clusters_coherence_lt_0.20': round(float((per_cluster < 0.20).mean()), 4),
        'member_to_centroid_mean': round(m2c, 4),
    }

    compare('lexnames', lex_labels, labels, report)
    compare('hypernym_depth2', anc2, labels, report)
    compare('hypernym_depth3', anc3, labels, report)

    # HEURISTIC cross-lexname classification (mechanical rule; NOT an
    # independent semantic judgment). Rule per multi-member cluster:
    #   single lexname                       -> agreement
    #   coherence >= 0.30 and (within-majority - cross) < 0.15
    #                                        -> cross_lexname_coherent
    #   coherence < 0.20 or gap >= 0.15      -> split_candidate
    #   otherwise                            -> borderline
    heur = Counter()
    heur_members = Counter()
    examples = {'cross_lexname_coherent': [], 'split_candidate': []}
    for g in np.where(counts >= 2)[0]:
        idx = np.where(labels == g)[0]
        lexs = [lex_labels[i] for i in idx]
        coh = within_cos(X, idx)
        if len(set(lexs)) == 1:
            cls = 'agreement'
        else:
            maj = Counter(lexs).most_common(1)[0][0]
            maj_idx = idx[[i for i, l in enumerate(lexs) if l == maj]]
            oth_idx = idx[[i for i, l in enumerate(lexs) if l != maj]]
            within = within_cos(X, maj_idx)
            cross = pair_cos(X, maj_idx, oth_idx)
            gap = (within - cross) if (within is not None and cross is not None) else 0.0
            if coh is not None and coh >= 0.30 and gap < 0.15:
                cls = 'cross_lexname_coherent'
            elif (coh is not None and coh < 0.20) or gap >= 0.15:
                cls = 'split_candidate'
            else:
                cls = 'borderline'
        heur[cls] += 1
        heur_members[cls] += len(idx)
        if cls in examples and len(examples[cls]) < 3:
            examples[cls].append({
                'cluster': int(g), 'size': int(len(idx)),
                'coherence': round(float(coh), 4) if coh is not None else None,
                'lexnames': dict(Counter(lexs)),
                'members': [{'synset_id': ids[i], 'lexname': lex_labels[i],
                             'text': texts[i]} for i in idx]})
    report['heuristic_classification'] = {
        'label': 'HEURISTIC — mechanical rule + author hand-read of a '
                 'small sample; not independent review',
        'rule': 'agreement: single lexname. cross_lexname_coherent: '
                'coherence>=0.30 and within-majority minus cross cosine '
                '<0.15. split_candidate: coherence<0.20 or gap>=0.15. '
                'else borderline.',
        'cluster_counts': dict(heur), 'member_counts': dict(heur_members),
        'examples': examples,
    }

    # Shared-headword false merges: clusters dominated by one headword
    # token across >=3 lexnames (word-form binding, not meaning).
    headwords = []
    for t in texts:
        m = HEADWORD_RE.match(t)
        headwords.append(m.group(1).lower().strip('.,') if m else '')
    false_merges = []
    for g in np.where(counts >= 4)[0]:
        idx = np.where(labels == g)[0]
        hw = Counter(headwords[i] for i in idx if headwords[i])
        if not hw:
            continue
        top, top_n = hw.most_common(1)[0]
        n_lex = len(set(lex_labels[i] for i in idx))
        if top_n / len(idx) >= 0.5 and n_lex >= 3:
            false_merges.append({
                'cluster': int(g), 'size': int(len(idx)), 'headword': top,
                'headword_share': round(top_n / len(idx), 2),
                'distinct_lexnames': n_lex,
                'coherence': round(float(within_cos(X, idx)), 4),
                'members': [{'synset_id': ids[i], 'lexname': lex_labels[i],
                             'text': texts[i]} for i in idx]})
    false_merges.sort(key=lambda d: (-d['distinct_lexnames'], -d['size']))
    report['shared_headword_candidates'] = {
        'definition': 'size>=4, one headword >=50% of members, >=3 lexnames',
        'count': len(false_merges), 'top_examples': false_merges[:6],
    }
    print(f'heuristic: {dict(heur)}; headword candidates: {len(false_merges)}',
          flush=True)

    OUT_JSON.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'written: {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
