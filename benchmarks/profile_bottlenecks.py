"""Bottleneck profile for batched-birch at WordNet 31k.

Runs (a) a cProfile of BirchTree.fit(wave=256) on the full WordNet
matrix and (b) a wave-size timing sweep on a 10k prefix, printing
the top cumulative functions and per-wave costs. Output is text for
BOTTLENECKS.md; no JSON (profiles are machine-specific).

Usage: python benchmarks/profile_bottlenecks.py
Env: BIRCH_WORDNET_VECTORS overrides the vectors path.
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent)]

VECTORS = Path(os.environ.get('BIRCH_WORDNET_VECTORS',
                              str(HERE / 'wordnet-vectors.npy')))


def main():
    from batched_birch import BirchConfig, BirchTree

    X = np.ascontiguousarray(np.load(VECTORS), dtype=np.float64)
    print(f'X {X.shape}', flush=True)

    pr = cProfile.Profile()
    pr.enable()
    bt = BirchTree(BirchConfig()).fit(X, wave_size=256)
    pr.disable()
    print(f'groups {bt.n_groups}', flush=True)
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(25)
    print(s.getvalue())

    print('--- wave sweep on 10k prefix (single timed fit each) ---')
    for w in (1, 32, 256, 1024):
        t0 = time.perf_counter()
        b = BirchTree(BirchConfig()).fit(X[:10000], wave_size=w)
        dt = time.perf_counter() - t0
        print(f'wave={w}: {dt:.2f}s groups={b.n_groups} '
              f'waves={-(-10000 // w)}', flush=True)


if __name__ == '__main__':
    main()
