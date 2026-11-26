"""Controlled benchmark suite: batched-birch vs scikit-learn Birch.

Method (see REPRODUCING.md for the full protocol):
  * One subprocess per arm. Imports and data loading happen OUTSIDE
    the timed region; only the fit call is timed.
  * 1 warmup fit on a small prefix (untimed), then repeated timed
    fits with a fresh model each rep. Reported: median + IQR of wall
    seconds. Never a single-run claim (except arms explicitly marked
    reps=1: wave_size=1 sequential diagnostics and the 1M attempt).
  * BLAS threads pinned to 1 via OMP_NUM_THREADS /
    OPENBLAS_NUM_THREADS / MKL_NUM_THREADS in the subprocess env
    (recorded in every fragment). A small sensitivity set runs at
    default threads (arm keys ending in ``__threads-default``).
  * Every fragment records the environment: CPU, RAM, OS, Python,
    NumPy / scikit-learn versions, BLAS config, thread env.
  * Per-arm JSON fragments in results/frags-2026-11-25/ are merged
    into results/benchmark-2026-11-25.json. Finished arms are
    skipped on relaunch (use --force to redo). Timeouts and
    MemoryErrors are recorded as results, not crashes.

Arms (semantics differ — read before comparing):
  ours              batched_birch BirchTree.fit, BirchConfig defaults
                    (cosine gate 0.5, dispersion 0.5, capacity 16),
                    waves of --wave (default 256). Arms with
                    ``__workersN`` in the key set BirchConfig.workers=N
                    (parallel scoring blocks; BLAS pinned to 1, labels
                    must match the workers=1 arm exactly).
  sklearn-fit       sklearn Birch(threshold=0.5, branching_factor=50,
                    n_clusters=None).fit(X). Threshold is a subcluster
                    RADIUS in Euclidean space, not a cosine gate.
  sklearn-partial   same Birch, partial_fit in waves of 256.
  sklearn-matched   sklearn fit with threshold tuned so the group
                    count at WordNet 31k approximates ours (0.865).
                    Granularity-matched, NOT semantics-matched.

Metrics per arm: wall s (median, IQR, all reps), items/s, peak RSS
MB (subprocess ru_maxrss where available), groups, singleton counts
and rates (per-group and per-item), cluster-size mean/median/p90/max,
pair-weighted intra-cluster cosine (coherence), mean member-to-
centroid cosine, and NMI + ARI vs ground truth where it exists
(synthetic blobs / duplicates). Labels come from BirchTree.labels_
(ours) or Birch.predict(X) (sklearn, both fit and partial_fit, so
the two sklearn arms are scored identically).

Usage:
    python benchmarks/bench_controlled.py            # run all arms
    python benchmarks/bench_controlled.py --list     # show arm keys
    python benchmarks/bench_controlled.py --only wordnet-31388
    python benchmarks/bench_controlled.py --merge-only

Requires: numpy, scikit-learn. WordNet vectors default to
benchmarks/wordnet-vectors.npy; override with BIRCH_WORDNET_VECTORS.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path[:0] = [str(PKG), str(HERE)]

RESULTS_DIR = PKG / 'results'
FRAG_DIR = RESULTS_DIR / 'frags-2026-11-25'
MERGED = RESULTS_DIR / 'benchmark-2026-11-25.json'
VECTORS = Path(os.environ.get('BIRCH_WORDNET_VECTORS',
                              str(HERE / 'wordnet-vectors.npy')))
SEED = 20261125
MATCHED_THRESHOLD = 0.865  # tuned at WordNet 31k: ~4.9k groups vs ours 4796

# ---------------------------------------------------------------- datasets

def _syn(dist, n, dim, **kw):
    return {'kind': 'synthetic', 'dist': dist, 'n': n, 'dim': dim,
            'seed': SEED, **kw}


DATASETS = {
    'wordnet-1000': {'kind': 'wordnet', 'n': 1000},
    'wordnet-31388': {'kind': 'wordnet', 'n': 31388},
    'syn-blobs-1000-d384': _syn('blobs', 1000, 384),
    'syn-blobs-10000-d384': _syn('blobs', 10000, 384),
    'syn-blobs-100000-d384': _syn('blobs', 100000, 384),
    'syn-blobs-10000-d128': _syn('blobs', 10000, 128),
    'syn-blobs-10000-d768': _syn('blobs', 10000, 768),
    'syn-uniform-10000-d384': _syn('uniform', 10000, 384),
    'syn-dup-10000-d384': _syn('duplicates', 10000, 384),
    'syn-blobs-1000000-d384': _syn('blobs', 1000000, 384),
}


def load_dataset(spec):
    """Return (X float64 C-contiguous, gt_labels or None). Untimed."""
    if spec['kind'] == 'wordnet':
        arr = np.load(VECTORS, mmap_mode='r')
        n = min(spec['n'], len(arr))
        X = np.ascontiguousarray(arr[:n], dtype=np.float64)
        del arr
        return X, None
    from gen_synthetic import generate
    X32, labels, _ = generate(spec['dist'], spec['n'], spec['dim'],
                              spec['seed'],
                              n_blobs=spec.get('n_blobs', 50),
                              spread=spec.get('spread', 0.8),
                              dup_frac=spec.get('dup_frac', 0.5))
    gt = None if (labels < 0).all() else labels
    return np.ascontiguousarray(X32, dtype=np.float64), gt


# ---------------------------------------------------------------- metrics

def cluster_metrics(X, labels, gt):
    labels = np.asarray(labels)
    n = len(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    groups = len(uniq)
    # Per-cluster vector sums (unit rows assumed; renormalize defensively
    # only for the coherence math, on a view-copy of sums).
    sums = np.zeros((groups, X.shape[1]), dtype=np.float64)
    # Map arbitrary label values to 0..groups-1.
    remap = np.searchsorted(uniq, labels)
    np.add.at(sums, remap, X)
    sq = np.einsum('ij,ij->i', sums, sums)
    pair_num = float((sq - counts).sum())          # 2 * sum of pairwise cos
    pair_den = float((counts * (counts - 1)).sum())
    coherence = pair_num / pair_den if pair_den else float('nan')
    cent = sums / counts[:, None]
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-300)
    m2c = float(np.einsum('ij,ij->i', X, cent[remap]).mean()) if n else float('nan')
    singletons = int((counts == 1).sum())
    out = {
        'groups': int(groups),
        'singleton_groups': singletons,
        'singleton_group_rate': round(singletons / groups, 4) if groups else None,
        'singleton_item_rate': round(singletons / n, 4) if n else None,
        'size_mean': round(float(counts.mean()), 3) if groups else None,
        'size_median': float(np.median(counts)) if groups else None,
        'size_p90': float(np.percentile(counts, 90)) if groups else None,
        'size_max': int(counts.max()) if groups else 0,
        'coherence_pair_weighted': round(coherence, 4),
        'member_to_centroid_mean': round(m2c, 4),
    }
    if gt is not None:
        from sklearn.metrics import (adjusted_rand_score,
                                     normalized_mutual_info_score)
        out['nmi_vs_gt'] = round(float(normalized_mutual_info_score(
            gt, labels, average_method='arithmetic')), 4)
        out['ari_vs_gt'] = round(float(adjusted_rand_score(gt, labels)), 4)
    return out


def peak_rss_mb():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: KiB. macOS: bytes. Windows: no resource module.
        return round(v / 1024.0, 1) if sys.platform != 'darwin' \
            else round(v / (1024.0 * 1024.0), 1)
    except Exception:
        try:
            import psutil
            return round(psutil.Process().memory_info().rss / 1e6, 1)
        except Exception:
            return None


def environment_record():
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except Exception:
        pass
    ram_gb = None
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemTotal'):
                ram_gb = round(int(line.split()[1]) / 1048576, 1)
                break
    except Exception:
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / 2**30, 1)
        except Exception:
            pass
    blas = ''
    try:
        buf = io.StringIO()
        np.show_config()
        blas = 'see numpy config (captured in driver log)'
    except Exception:
        pass
    try:
        import sklearn
        skv = sklearn.__version__
    except Exception:
        skv = None
    return {
        'cpu': cpu, 'cpu_count': os.cpu_count(), 'ram_gb': ram_gb,
        'os': platform.platform(), 'python': sys.version.split()[0],
        'numpy': np.__version__, 'sklearn': skv,
        'threads_env': {k: os.environ.get(k) for k in
                        ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'MKL_NUM_THREADS')},
    }


# ---------------------------------------------------------------- one arm

def fit_once(impl, X, wave, threshold, workers=1):
    """Fit only (timed by caller). Returns a fitted handle."""
    if impl == 'ours':
        from batched_birch import BirchConfig, BirchTree
        return BirchTree(BirchConfig(workers=workers)).fit(X, wave_size=wave)
    from sklearn.cluster import Birch
    model = Birch(threshold=threshold, branching_factor=50, n_clusters=None)
    if impl == 'sklearn-fit':
        model.fit(X)
    elif impl == 'sklearn-partial':
        for s in range(0, len(X), 256):
            model.partial_fit(X[s:s + 256])
    else:
        raise ValueError(impl)
    return model


def labels_of(impl, handle, X):
    """Labels for metrics (UNTIMED). sklearn: nearest-subcluster-center
    assignment via predict — the same scoring for fit and partial_fit.
    Note predict cost grows as n x groups x dim and is reported
    separately as predict_s; it is never inside wall_s."""
    if impl == 'ours':
        labels = handle.labels_
        handle.close()  # release the scoring pool (workers > 1)
        return labels
    return handle.predict(X)


def run_one(arm):
    """Executed inside the subprocess. Returns the fragment dict."""
    spec = DATASETS[arm['dataset']]
    frag = {'arm': arm, 'dataset_spec': spec,
            'environment': environment_record(), 'status': 'ok'}
    t_load = time.perf_counter()
    X, gt = load_dataset(spec)
    frag['load_s'] = round(time.perf_counter() - t_load, 2)
    frag['n'] = int(len(X))
    frag['dim'] = int(X.shape[1])
    # Warmup on a prefix (untimed): JIT/BLAS/page-cache warm.
    warm_n = min(len(X), 2000)
    try:
        fit_once(arm['impl'], X[:warm_n], arm.get('wave', 256),
                 arm.get('threshold', 0.5), arm.get('workers', 1))
    except Exception as exc:  # warmup failure is still informative
        frag['warmup_error'] = f'{type(exc).__name__}: {exc}'
    walls, hashes, labels = [], [], None
    predict_s = None
    # sklearn predict is O(n x groups x dim); at large n compute labels
    # once (last rep) instead of per rep. sklearn Birch is
    # deterministic for fixed input order, so one labeling suffices;
    # reps_labels_identical then reflects only the labeled reps.
    label_each_rep = arm['impl'] == 'ours' or len(X) <= 40000
    for rep in range(arm['reps']):
        t0 = time.perf_counter()
        handle = fit_once(arm['impl'], X, arm.get('wave', 256),
                          arm.get('threshold', 0.5), arm.get('workers', 1))
        walls.append(time.perf_counter() - t0)
        if label_each_rep or rep == arm['reps'] - 1:
            tp = time.perf_counter()
            labels = labels_of(arm['impl'], handle, X)
            predict_s = round(time.perf_counter() - tp, 3)
            hashes.append(hashlib.sha256(
                np.asarray(labels).tobytes()).hexdigest()[:16])
    walls_sorted = sorted(walls)
    med = float(np.median(walls))
    if len(walls) >= 3:
        q1, q3 = np.percentile(walls, [25, 75])
        iqr = float(q3 - q1)
    else:
        iqr = float(walls_sorted[-1] - walls_sorted[0])
    frag.update({
        'wall_s_reps': [round(w, 3) for w in walls],
        'wall_s_median': round(med, 3),
        'wall_s_iqr': round(iqr, 3),
        'items_per_s_median': round(len(X) / med, 1) if med else None,
        'peak_rss_mb': peak_rss_mb(),
        'labels_sha16': hashes[-1],
        'reps_labels_identical': len(set(hashes)) == 1,
        'predict_s_last_rep': predict_s,
    })
    frag.update(cluster_metrics(X, labels, gt))
    return frag


# ---------------------------------------------------------------- arm matrix

def arm_key(a):
    k = f"{a['dataset']}__{a['impl']}"
    if a['impl'] == 'ours':
        k += f"__w{a.get('wave', 256)}"
        if a.get('workers', 1) != 1:
            k += f"__workers{a['workers']}"
    if 'threshold' in a and a['impl'] != 'ours':
        k += f"__t{a['threshold']}"
    if a.get('threads') == 'default':
        k += '__threads-default'
    return k


def build_matrix():
    arms = []

    def add(dataset, impl, reps=3, **kw):
        arms.append({'dataset': dataset, 'impl': impl, 'reps': reps, **kw})

    # 1. WordNet anchor (before/after vs old single-run results).
    for ds in ('wordnet-1000', 'wordnet-31388'):
        add(ds, 'ours', wave=256)
        add(ds, 'sklearn-fit', threshold=0.5)
        add(ds, 'sklearn-partial', threshold=0.5)
    add('wordnet-31388', 'sklearn-fit', threshold=MATCHED_THRESHOLD)
    # Thread sensitivity (default threads) at 31k.
    add('wordnet-31388', 'ours', reps=1, wave=256, threads='default')
    add('wordnet-31388', 'sklearn-fit', reps=1, threshold=0.5, threads='default')
    # 2. Synthetic scale ladder (clustered blobs, dim 384).
    for ds, reps in (('syn-blobs-1000-d384', 3), ('syn-blobs-10000-d384', 3),
                     ('syn-blobs-100000-d384', 2)):
        add(ds, 'ours', reps=reps, wave=256)
        add(ds, 'sklearn-fit', reps=reps, threshold=0.5)
        add(ds, 'sklearn-partial', reps=reps, threshold=0.5)
    # Thread sensitivity (default threads) at 100k, ours only.
    add('syn-blobs-100000-d384', 'ours', reps=1, wave=256, threads='default')
    add('syn-blobs-1000000-d384', 'ours', reps=1, wave=256, threads='default')
    # 2b. Mode B parallel scoring blocks (ours only, BLAS pinned to 1
    # per worker — the harness default). Labels must match the
    # workers=1 arms exactly (labels_sha16 in the frags).
    for ds, reps in (('syn-blobs-1000-d384', 3), ('syn-blobs-100000-d384', 2)):
        add(ds, 'ours', reps=reps, wave=256, workers=2)
        add(ds, 'ours', reps=reps, wave=256, workers=4)
    # 3. Wave-size sweep (ours only). wave=1 is sequential semantics.
    for ds in ('wordnet-31388', 'syn-blobs-10000-d384'):
        add(ds, 'ours', reps=1, wave=1)
        add(ds, 'ours', reps=3, wave=32)
        add(ds, 'ours', reps=3, wave=1024)
    # 4. Dim sweep at 10k clustered (384 covered above).
    for ds in ('syn-blobs-10000-d128', 'syn-blobs-10000-d768'):
        add(ds, 'ours', wave=256)
        add(ds, 'sklearn-fit', threshold=0.5)
        add(ds, 'sklearn-partial', threshold=0.5)
    # 5. Distributions at 10k x 384.
    for ds in ('syn-uniform-10000-d384', 'syn-dup-10000-d384'):
        add(ds, 'ours', wave=256)
        add(ds, 'sklearn-fit', threshold=0.5)
        add(ds, 'sklearn-partial', threshold=0.5)
    # 6. 1M attempt (expected infeasible on the VM; host arm).
    add('syn-blobs-1000000-d384', 'ours', reps=1, wave=256)
    add('syn-blobs-1000000-d384', 'sklearn-fit', reps=1, threshold=0.5)
    return arms


# ---------------------------------------------------------------- driver

def merge():
    merged = {'generated': '2026-11-25', 'arms': {}}
    for frag_path in sorted(FRAG_DIR.glob('*.json')):
        try:
            frag = json.loads(frag_path.read_text())
        except Exception:
            continue
        merged['arms'][frag_path.stem] = frag
    MERGED.write_text(json.dumps(merged, indent=2))
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run-one', nargs=2, metavar=('KEY', 'ARM_JSON'),
                    help='internal: run one arm in this process')
    ap.add_argument('--only', default='', help='substring filter on arm keys')
    ap.add_argument('--timeout', type=int, default=900, help='per-arm cap (s)')
    ap.add_argument('--force', action='store_true', help='redo finished arms')
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--merge-only', action='store_true')
    args = ap.parse_args()

    if args.run_one:
        key, arm_json = args.run_one
        # Memory guard (Unix): fail with MemoryError instead of OOM-kill.
        try:
            import resource
            limit = 5_500 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except Exception:
            pass
        try:
            frag = run_one(json.loads(arm_json))
        except MemoryError as exc:
            frag = {'arm': json.loads(arm_json), 'status': 'memory_error',
                    'error': f'MemoryError: {exc}',
                    'environment': environment_record()}
        except Exception as exc:
            frag = {'arm': json.loads(arm_json), 'status': 'error',
                    'error': f'{type(exc).__name__}: {exc}'[:2000],
                    'environment': environment_record()}
        FRAG_DIR.mkdir(parents=True, exist_ok=True)
        (FRAG_DIR / f'{key}.json').write_text(json.dumps(frag, indent=2))
        print(json.dumps({'key': key, 'status': frag['status'],
                          'wall': frag.get('wall_s_median'),
                          'groups': frag.get('groups')}))
        return

    if args.merge_only:
        m = merge()
        print(f'merged {len(m["arms"])} arms -> {MERGED}')
        return

    arms = build_matrix()
    if args.list:
        for a in arms:
            print(arm_key(a), a)
        return

    FRAG_DIR.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ)
    base_env.setdefault('ORT_DISABLE_TELEMETRY', '1')
    for a in arms:
        key = arm_key(a)
        if args.only and args.only not in key:
            continue
        frag_path = FRAG_DIR / f'{key}.json'
        if frag_path.exists() and not args.force:
            print(f'skip {key} (done)', flush=True)
            continue
        env = dict(base_env)
        if a.get('threads') != 'default':
            env.update({'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1'})
        timeout = args.timeout
        if a.get('wave') == 1 and a['dataset'] == 'wordnet-31388':
            timeout = min(timeout, 300)  # sequential diagnostic cap
        print(f'run {key} (timeout {timeout}s) ...', flush=True)
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, __file__, '--run-one', key, json.dumps(a)],
                capture_output=True, text=True, timeout=timeout, env=env)
            if proc.returncode != 0 or not frag_path.exists():
                frag = {'arm': a, 'status': 'error',
                        'error': (proc.stderr or '')[-2000:],
                        'environment': environment_record()}
                frag_path.write_text(json.dumps(frag, indent=2))
        except subprocess.TimeoutExpired:
            frag = {'arm': a, 'status': 'timeout',
                    'error': f'timeout after {timeout}s',
                    'environment': environment_record()}
            frag_path.write_text(json.dumps(frag, indent=2))
        frag = json.loads(frag_path.read_text())
        print(f"  -> {frag.get('status')} wall={frag.get('wall_s_median')} "
              f"groups={frag.get('groups')} ({time.monotonic() - t0:.0f}s)",
              flush=True)
        merge()
    merge()
    print(f'results: {MERGED}')


if __name__ == '__main__':
    main()
