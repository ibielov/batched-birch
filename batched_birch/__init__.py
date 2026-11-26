"""batched-birch: batched (set-wise) BIRCH clustering over embedding
vectors, with numerically stable BETULA-style (n, mu, S) cluster
features (Welford/Chan updates; dispersion = S/n). Numpy-only core;
embedder/store/concepts are opt-out adapters around it. See
batched_birch.core for the full semantics (wave routing,
within-wave consolidation, validation, unit-norm policy).
"""
from .core import BirchConfig, BirchTree, CFTable, WaveResult, place_wave, split_group

__version__ = '0.1.0'

__all__ = [
    'BirchConfig', 'CFTable', 'WaveResult', 'place_wave', 'split_group',
    'BirchTree',
]
