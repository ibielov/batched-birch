"""Batched BIRCH core — public-library shape.

Cluster features are the numerically stable BETULA-style (n, mu, S)
triple — count, mean vector, and sum of squared deviations from the
mean — maintained with Welford point updates and Chan group merges,
instead of the classic (N, LS, SS) with dispersion =
SS/N - ||LS||^2/N^2 (which catastrophically cancels for data far
from the origin). Dispersion is S/n, computed directly. Routing,
similarity, acceptability, tie-breaking, wave semantics and the
split rule are unaffected by the CF representation.

The core knows vectors, CF statistics and a flat group tree — nothing
about embedders, databases, concepts or proposals. Those are opt-out
adapters around the core, never inside it.

Wave semantics: same-wave items route against ONE pinned snapshot
(simultaneous/joint). Items do not see each other's tentative changes,
so a whole wave is scored set-wise (one matmul; with workers > 1, one
matmul per worker block — see below), acceptability and winner
selection are vectorized, and the joint CF merge is a vectorized
Chan merge of per-group (count, mean, S) segment statistics.
The only sequential residue is splitting overflowed groups.

Parallel scoring (BirchConfig.workers > 1): the centroid (group)
rows are partitioned into contiguous blocks, one per worker; each
worker computes its block's scores against the same pinned snapshot
and returns a local top-k; the merge orders the union by (score
desc, group index asc) and truncates to k — the exact global top-k,
identical to the serial shortlist, ties included. Partitioning is
over group rows ONLY, never over the feature dimension, so every
score remains the identical dot product. The full (items, groups)
score matrix is never materialized in this mode. workers=1 (the
default) is exactly the serial path and spawns no threads. NumPy
matmul releases the GIL, so blocks run genuinely in parallel; the
Python-level merge and everything downstream stay serial. With
workers > 1, pin BLAS to one thread per worker
(OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS = 1,
set before importing NumPy) — W workers x T BLAS threads each
oversubscribes the machine and can be dramatically slower.

Within-wave consolidation (new-group formation): items that win no
existing group — including every item of a first wave on an empty
tree — are consolidated among themselves set-wise before any new
group is committed. Rounds over the pinned pairwise similarities of
the unassigned pool: the lowest-index unconsolidated item seeds a
round; pool items with cosine >= min_similarity to the seed are
ranked by similarity to the seed (ties: lowest index); the largest
prefix of that ranking whose joint CF satisfies max_children and
max_dispersion (sequential Welford prefix updates) becomes one new group,
committed as a single CF row. Rounds repeat until the pool is
empty; an item similar to nothing stays a singleton. This is not
per-item insertion against a growing tree: membership is decided
from pinned similarity rows and committed set-wise. It also does
not weaken pinned-snapshot routing — an item that fails every
existing group cannot join one through a same-wave neighbour that
succeeded; it can only consolidate with other unassigned items.

Unit-norm policy: inputs are NOT required to be pre-normalized.
Every public entry point (place_wave, split_group, BirchTree.fit /
partial_fit) validates its input and works on a unit-normalized
float64 COPY; the caller's array is never mutated, and similarity
is cosine. Zero rows are rejected (they have no direction).
CFTable.append_singletons / append_groups are raw builders: they
store vectors exactly as given (validated finite and non-zero)
and expect unit vectors.

Validation: BirchConfig ranges are checked at construction and
raise ValueError (0 < min_similarity <= 1, max_dispersion > 0,
max_children >= 1, candidate_count >= 1, workers >= 1). Inputs
must be 2-D, finite (no NaN/inf), free of zero rows, and
dimension-consistent with the tree; violations raise ValueError.

fit() RESETS the model (fresh tree, fresh history) before fitting;
partial_fit() is incremental. fit(X, wave_size=1) reproduces
classic per-item sequential insertion exactly (each item sees all
previous items' groups; a one-item pool is a singleton). Larger
waves use pinned-snapshot routing plus consolidation and may
produce a different partition — no subset/superset relationship
between wave sizes is guaranteed — but every wave size satisfies
the same invariants: each item in exactly one group, CF statistics
consistent with members, and group constraints enforced once split
resolution has run.

After each wave's joint merge, every touched group (pre-existing
groups that received items, plus newly formed groups) is checked
against max_children / max_dispersion. Violators are reported in
WaveResult.splits_needed — place_wave itself does not split;
BirchTree resolves them via split_group until no group violates.
Split resolution terminates: every split produces two non-empty,
strictly smaller groups (identical-vector groups fall back to an
index split), so recursion bottoms out at singletons, which always
satisfy the constraints.

Complexity note: consolidation is O(pool^2 * dim) worst-case time
and O(pool * dim) memory (one similarity row per round). The pool
is bounded by the wave size in BirchTree use (default 256).

Determinism: fixed input order, explicit tie-breaking everywhere
(lowest after-dispersion, then highest similarity, then lowest group
index). No randomness.

Only numpy is required. pandas is accepted at the edges by callers
(frames convert to arrays); the core itself is array-only.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    'BirchConfig', 'CFTable', 'WaveResult', 'place_wave', 'split_group',
    'BirchTree',
]


@dataclass(frozen=True)
class BirchConfig:
    min_similarity: float = 0.5        # cosine gate (unit vectors)
    max_dispersion: float = 0.5        # mean squared radius limit
    max_children: int = 16             # leaf capacity
    candidate_count: int = 5           # shortlist size per item
    workers: int = 1                   # parallel scoring blocks (1 = serial)

    def __post_init__(self):
        if not 0 < self.min_similarity <= 1:
            raise ValueError(
                f'min_similarity must satisfy 0 < min_similarity <= 1, '
                f'got {self.min_similarity!r}')
        if not self.max_dispersion > 0:
            raise ValueError(
                f'max_dispersion must be > 0, got {self.max_dispersion!r}')
        if not self.max_children >= 1:
            raise ValueError(
                f'max_children must be >= 1, got {self.max_children!r}')
        if not self.candidate_count >= 1:
            raise ValueError(
                f'candidate_count must be >= 1, got {self.candidate_count!r}')
        if not self.workers >= 1 or int(self.workers) != self.workers:
            raise ValueError(
                f'workers must be an integer >= 1, got {self.workers!r}')


def _prepare_X(X, expected_dim: int | None = None) -> np.ndarray:
    """Validate X and return a unit-normalized float64 COPY.

    Raises ValueError unless X is a 2-D array with at least one
    column, all entries finite, no zero rows, and (when
    expected_dim is given) the expected number of columns. The
    caller's array is never mutated. Row normalization uses a
    max-abs prescale so extreme magnitudes cannot overflow/underflow
    the norm.
    """
    arr = np.array(X, dtype=np.float64)  # always a fresh copy
    if arr.ndim != 2:
        raise ValueError('X must be a 2-D array (items, dim)')
    if arr.shape[1] == 0:
        raise ValueError('X must have at least one column (dim >= 1)')
    if expected_dim is not None and arr.shape[1] != expected_dim:
        raise ValueError(
            f'X dimension {arr.shape[1]} does not match expected '
            f'dimension {expected_dim}')
    if arr.shape[0] == 0:
        return arr
    if not np.isfinite(arr).all():
        raise ValueError('X must contain only finite values (no NaN/inf)')
    scale = np.max(np.abs(arr), axis=1)
    if (scale == 0).any():
        raise ValueError('X must not contain zero vectors')
    arr /= scale[:, None]
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    return arr


def _validate_rows(X, expected_dim: int) -> np.ndarray:
    """Validate raw-builder input (no normalization). Returns float64 view."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError('X must be a 2-D array (items, dim)')
    if arr.shape[1] != expected_dim:
        raise ValueError(
            f'X dimension {arr.shape[1]} does not match tree dimension '
            f'{expected_dim}')
    if arr.shape[0] and not np.isfinite(arr).all():
        raise ValueError('X must contain only finite values (no NaN/inf)')
    if arr.shape[0] and (np.max(np.abs(arr), axis=1) == 0).any():
        raise ValueError('X must not contain zero vectors')
    return arr


@dataclass
class CFTable:
    """The whole tree state as typed columns (one row per group).

    Stable cluster feature (BETULA-style), one row per group:
    n:    member count                          (int64)
    mu:   group mean vector                     (float64, (groups, dim))
    S:    sum of squared deviations from mu     (float64)
    Dispersion (mean squared radius) is S / n — computed directly,
    with no SS - ||LS||^2/N cancellation. Point adds use Welford's
    update, group merges use Chan's parallel formula; both keep
    every intermediate on the scale of the data's spread.
    parent: parent group index or -1 for roots (flat tree: always -1
            in this version; kept for forward compatibility)
    ids:  external group ids, aligned to rows
    """
    n: np.ndarray
    mu: np.ndarray
    S: np.ndarray
    parent: np.ndarray
    ids: list

    @classmethod
    def empty(cls, dim: int) -> 'CFTable':
        if dim < 1:
            raise ValueError(f'dim must be >= 1, got {dim!r}')
        return cls(
            n=np.zeros(0, dtype=np.int64),
            mu=np.zeros((0, dim), dtype=np.float64),
            S=np.zeros(0, dtype=np.float64),
            parent=np.zeros(0, dtype=np.int64),
            ids=[],
        )

    def __len__(self) -> int:
        return len(self.ids)

    def copy(self) -> 'CFTable':
        return CFTable(self.n.copy(), self.mu.copy(), self.S.copy(),
                       self.parent.copy(), list(self.ids))

    def centroids(self) -> np.ndarray:
        """Unit-normalized group centroids, shape (groups, dim)."""
        if len(self) == 0:
            return self.mu.copy()
        c = self.mu
        norms = np.linalg.norm(c, axis=1, keepdims=True)
        return c / np.where(norms == 0, 1.0, norms)

    def dispersion(self) -> np.ndarray:
        """Mean squared radius per group (S / n), vectorized."""
        if len(self) == 0:
            return self.S.copy()
        return self.S / self.n

    def dispersion_rows(self, rows) -> np.ndarray:
        """Dispersion for the given group rows only.

        Same per-row arithmetic as dispersion() restricted to those
        rows (identical values), without touching the rest of the
        tree. Used by split resolution and by the per-wave split
        check in _place_wave_impl, which only ever need the
        dispersion of the groups they touched.
        """
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == 0:
            return np.zeros(0, dtype=np.float64)
        return self.S[rows] / self.n[rows]

    def append_singletons(self, X: np.ndarray, id_prefix: str = 'g') -> np.ndarray:
        """Append one singleton group per row of X; return new indices.

        Raw builder: vectors are stored exactly as given (expected
        unit vectors); validated 2-D, finite, non-zero, and
        dimension-consistent with the tree.
        """
        X = _validate_rows(X, self.mu.shape[1])
        start = len(self)
        m = len(X)
        if m == 0:
            return np.zeros(0, dtype=np.int64)
        self.n = np.concatenate([self.n, np.ones(m, dtype=np.int64)])
        self.mu = np.vstack([self.mu, X])
        self.S = np.concatenate([self.S, np.zeros(m, dtype=np.float64)])
        self.parent = np.concatenate(
            [self.parent, np.full(m, -1, dtype=np.int64)])
        self.ids.extend(f'{id_prefix}{start + i}' for i in range(m))
        return np.arange(start, start + m, dtype=np.int64)

    def append_groups(self, X: np.ndarray, groups, id_prefix: str = 'g') -> np.ndarray:
        """Append one group per member-index list; return new indices.

        groups: sequence of non-empty index arrays into the rows of
        X. CF statistics are segment sums over the members. Raw
        builder like append_singletons: vectors stored as given
        (expected unit vectors), validated finite/non-zero.
        """
        X = _validate_rows(X, self.mu.shape[1])
        groups = [np.asarray(g, dtype=np.int64) for g in groups]
        if any(len(g) == 0 for g in groups):
            raise ValueError('groups must not contain empty member lists')
        start = len(self)
        m = len(groups)
        if m == 0:
            return np.zeros(0, dtype=np.int64)
        rows = np.concatenate(groups)
        item_group = np.concatenate(
            [np.full(len(g), i, dtype=np.int64) for i, g in enumerate(groups)])
        n_new = np.bincount(item_group, minlength=m).astype(np.int64)
        # Two-pass stable CF: segment sums -> group means, then S
        # from deviations around those means (no SS - ||LS||^2/N).
        sums_new = np.zeros((m, X.shape[1]), dtype=np.float64)
        np.add.at(sums_new, item_group, X[rows])
        mu_new = sums_new / n_new[:, None]
        dev = X[rows] - mu_new[item_group]
        S_new = np.zeros(m, dtype=np.float64)
        np.add.at(S_new, item_group, np.einsum('ij,ij->i', dev, dev))
        self.n = np.concatenate([self.n, n_new])
        self.mu = np.vstack([self.mu, mu_new])
        self.S = np.concatenate([self.S, S_new])
        self.parent = np.concatenate(
            [self.parent, np.full(m, -1, dtype=np.int64)])
        self.ids.extend(f'{id_prefix}{start + i}' for i in range(m))
        return np.arange(start, start + m, dtype=np.int64)


@dataclass
class WaveResult:
    assignment: np.ndarray        # group row index per item (post-merge tree)
    new_groups: np.ndarray        # item indices assigned to groups created this wave
    splits_needed: list           # group rows to split (sequential residue)
    tree: CFTable                 # updated tree (merged + new groups)
    deferred: list = field(default_factory=list)  # {item, cause}


def _shortlist(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k group indices per item: score desc, group index asc.

    Fast path: argpartition selects the top-k set, then the k are
    ordered by (score desc, index asc). argpartition alone is not
    tie-safe at the cutoff — among scores exactly equal to the k-th
    score it keeps an arbitrary subset — so rows where an outside
    element ties the boundary score fall back to a full stable
    argsort. Result is fully deterministic on any platform.
    """
    items, groups = scores.shape
    if k >= groups:
        return np.argsort(-scores, axis=1, kind='stable')
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    # Order the k by index asc, then stable by score desc.
    by_index = np.argsort(part, axis=1, kind='stable')
    part = np.take_along_axis(part, by_index, axis=1)
    part_scores = np.take_along_axis(part_scores, by_index, axis=1)
    by_score = np.argsort(-part_scores, axis=1, kind='stable')
    shortlist = np.take_along_axis(part, by_score, axis=1)
    # Boundary ties: an element outside the partition with score equal
    # to the k-th score could displace a higher-index tied element.
    boundary = part_scores.min(axis=1)
    in_part = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(in_part, part, True, axis=1)
    tied_rows = ((scores == boundary[:, None]) & ~in_part).any(axis=1)
    if tied_rows.any():
        full = np.argsort(-scores[tied_rows], axis=1, kind='stable')[:, :k]
        shortlist[tied_rows] = full
    return shortlist


def _block_bounds(groups: int, workers: int) -> list:
    """Contiguous [start, stop) group-row blocks, sizes differing by
    at most 1 (earlier blocks take the remainder). Empty blocks
    (workers > groups) are dropped."""
    base, extra = divmod(groups, workers)
    bounds = []
    start = 0
    for i in range(workers):
        stop = start + base + (1 if i < extra else 0)
        if stop > start:
            bounds.append((start, stop))
        start = stop
    return bounds


def _score_block(Xn: np.ndarray, C: np.ndarray, start: int, stop: int,
                 k: int) -> tuple:
    """One worker's block: block scores + local top-k.

    Returns (indices, scores), each (items, min(k, block width)),
    with indices already offset to global group rows. Runs in a
    worker thread; reads Xn/C only (the pinned snapshot).
    """
    block_scores = Xn @ C[start:stop].T
    local = _shortlist(block_scores, min(k, stop - start))
    cand_scores = np.take_along_axis(block_scores, local, axis=1)
    return local + start, cand_scores


def _shortlist_blockwise(Xn: np.ndarray, C: np.ndarray, k: int,
                         workers: int, executor=None) -> tuple:
    """Exact global top-k shortlist via per-block scoring (Mode B).

    Partitions the centroid rows into `workers` contiguous blocks
    (over GROUPS only — never over the feature dimension, so every
    score is the identical dot product), scores each block in a
    worker thread, and merges the per-block top-k lists into the
    exact global top-k ordered by (score desc, group index asc) —
    the same result _shortlist(Xn @ C.T, k) produces, ties included
    (the global top-k is always a subset of the union of per-block
    top-k). The full (items, groups) score matrix is never
    materialized. Returns (shortlist, cand_scores), each (items, k).

    executor: a ThreadPoolExecutor to reuse, or None for a
    short-lived pool scoped to this call. Requires C.shape[0] >= 1
    (guaranteed by the caller) and 1 <= k <= C.shape[0].
    """
    bounds = _block_bounds(C.shape[0], workers)
    if executor is None:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(
                lambda b: _score_block(Xn, C, b[0], b[1], k), bounds))
    else:
        parts = list(executor.map(
            lambda b: _score_block(Xn, C, b[0], b[1], k), bounds))
    idx = np.concatenate([p[0] for p in parts], axis=1)
    sc = np.concatenate([p[1] for p in parts], axis=1)
    # Order by group index asc, then stable by score desc — the same
    # (score desc, index asc) order _shortlist produces.
    by_index = np.argsort(idx, axis=1, kind='stable')
    idx = np.take_along_axis(idx, by_index, axis=1)
    sc = np.take_along_axis(sc, by_index, axis=1)
    by_score = np.argsort(-sc, axis=1, kind='stable')
    idx = np.take_along_axis(idx, by_score, axis=1)
    sc = np.take_along_axis(sc, by_score, axis=1)
    return idx[:, :k], sc[:, :k]


def _winners(acceptable: np.ndarray, cand_disp: np.ndarray,
             cand_scores: np.ndarray, shortlist: np.ndarray) -> np.ndarray:
    """Winning candidate slot per item, vectorized with explicit ties.

    Order: lowest after-dispersion, then highest similarity, then
    lowest group index. Returns slot index per item, -1 if none.
    """
    items = acceptable.shape[0]
    rows = np.arange(items)
    masked_disp = np.where(acceptable, cand_disp, np.inf)
    best_disp = masked_disp.min(axis=1)
    has = np.isfinite(best_disp)
    # Among acceptable candidates tied on dispersion, highest similarity.
    tied = acceptable & (cand_disp == best_disp[:, None])
    masked_sim = np.where(tied, cand_scores, -np.inf)
    best_sim = masked_sim.max(axis=1)
    tied2 = tied & (cand_scores == best_sim[:, None])
    # Among those, lowest group index.
    big = np.iinfo(np.int64).max
    masked_idx = np.where(tied2, shortlist, big)
    slot = np.argmin(masked_idx, axis=1)
    return np.where(has, slot, -1)


def _consolidate(Xn: np.ndarray, pool: np.ndarray, cfg: BirchConfig) -> list:
    """Form new groups from unassigned pool items, set-wise.

    Xn: prepared (unit) wave vectors. pool: wave-item indices that
    won no existing group, ascending. Returns a list of index arrays
    (into Xn rows), one per new group, in formation order. See the
    module docstring for the round/prefix rule. Every returned
    group satisfies max_children and max_dispersion by construction
    (the singleton prefix is always valid, so rounds always make
    progress and the loop terminates).
    """
    P = Xn[pool]
    remaining = np.arange(len(pool), dtype=np.int64)  # positions in P
    groups = []
    while len(remaining):
        seed_pos = remaining[0]
        sims = P[remaining] @ P[seed_pos]            # pinned similarity row
        cand_mask = sims >= cfg.min_similarity
        cand_mask[0] = True                          # seed always included
        cand_pos = np.where(cand_mask)[0]            # positions in remaining
        # Rank: similarity desc, then pool position asc; seed first.
        order = np.lexsort((remaining[cand_pos], -sims[cand_pos]))
        cand_pos = cand_pos[order]
        seed_at = int(np.where(cand_pos == 0)[0][0])
        if seed_at:
            cand_pos = np.concatenate(
                [[0], np.delete(cand_pos, seed_at)])
        # Largest prefix whose joint CF satisfies the constraints.
        # Prefix dispersions via sequential Welford updates over the
        # ranked candidates: prefix S has no cancellation-free
        # closed form from cumulative sums (the SS - ||LS||^2/N form
        # is exactly what this variant removes), so each prefix's
        # (mu, S) is updated point by point. The loop is bounded by
        # the candidate count (<= pool size <= wave size in
        # BirchTree use) and preserves the exact prefix-selection
        # logic below.
        V = P[remaining[cand_pos]]
        counts = np.arange(1, len(V) + 1, dtype=np.float64)
        disp = np.empty(len(V), dtype=np.float64)
        mu_run = np.zeros(V.shape[1], dtype=np.float64)
        S_run = 0.0
        for i in range(len(V)):
            delta = V[i] - mu_run
            mu_run = mu_run + delta / (i + 1)
            S_run = S_run + float(delta @ delta) * i / (i + 1)
            disp[i] = S_run / (i + 1)
        ok = (disp <= cfg.max_dispersion) & (counts <= cfg.max_children)
        ok[0] = True                                 # singleton always valid
        k = int(np.max(np.where(ok)[0])) + 1
        groups.append(pool[remaining[cand_pos[:k]]])
        keep = np.ones(len(remaining), dtype=bool)
        keep[cand_pos[:k]] = False
        remaining = remaining[keep]
    return groups


def _place_wave_impl(Xn: np.ndarray, tree: CFTable, cfg: BirchConfig,
                     executor=None) -> WaveResult:
    """place_wave on prepared input (unit float64, dim checked).

    executor: optional ThreadPoolExecutor reused for parallel
    scoring when cfg.workers > 1 (BirchTree passes its persistent
    pool; None means a short-lived pool per call). Ignored when
    cfg.workers == 1 — the serial path spawns no threads.
    """
    items, dim = Xn.shape
    if items == 0:
        return WaveResult(np.zeros(0, dtype=np.int64),
                          np.zeros(0, dtype=np.int64), [], tree.copy())

    updated = tree.copy()
    assignment = np.full(items, -1, dtype=np.int64)

    if len(tree) > 0:
        C = tree.centroids()                              # (groups, dim)
        k = min(cfg.candidate_count, len(tree))
        if cfg.workers > 1:
            # Parallel scoring blocks: exact global top-k without
            # materializing the (items, groups) score matrix.
            shortlist, cand_scores = _shortlist_blockwise(
                Xn, C, k, int(cfg.workers), executor)
        else:
            scores = Xn @ C.T                             # (items, groups) ONE matmul
            shortlist = _shortlist(scores, k)             # (items, k)
            cand_scores = np.take_along_axis(scores, shortlist, axis=1)

        # Acceptability, vectorized over (item, candidate) pairs.
        # After-insert dispersion from stable CF arithmetic
        # (Welford point add):
        #   S' = S + ||x - mu||^2 * n/(n+1) ; dispersion' = S'/(n+1)
        cand_n = tree.n[shortlist] + 1                                  # (items, k)
        diff = Xn[:, None, :] - tree.mu[shortlist]                      # (items, k, dim)
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        cand_disp = (tree.S[shortlist]
                     + d2 * tree.n[shortlist] / cand_n) / cand_n
        acceptable = (
            (cand_scores >= cfg.min_similarity)
            & (cand_disp <= cfg.max_dispersion)
            & (cand_n <= cfg.max_children))

        slot = _winners(acceptable, cand_disp, cand_scores, shortlist)
        rows = np.arange(items)
        has_winner = slot >= 0
        assignment = np.where(
            has_winner, shortlist[rows, np.maximum(slot, 0)], -1).astype(np.int64)

    # Joint reconciliation: items landing in one existing group merge
    # their CFs together. Same-wave items never saw each other; the
    # merge is simultaneous. Stable form: per-group (count, mean, S)
    # of the newly placed points via segment ops (S from deviations
    # around the new points' own mean), then a vectorized Chan merge
    # with each touched group's existing CF.
    placed = assignment >= 0
    if placed.any():
        G = len(updated)
        tgt = assignment[placed]
        counts_new = np.bincount(tgt, minlength=G).astype(np.float64)
        sums_new = np.zeros((G, dim), dtype=np.float64)
        np.add.at(sums_new, tgt, Xn[placed])
        hit = counts_new > 0
        mu_new = np.zeros((G, dim), dtype=np.float64)
        mu_new[hit] = sums_new[hit] / counts_new[hit, None]
        dev = Xn[placed] - mu_new[tgt]
        S_new = np.zeros(G, dtype=np.float64)
        np.add.at(S_new, tgt, np.einsum('ij,ij->i', dev, dev))
        idx = np.where(hit)[0]
        n_old = updated.n[idx].astype(np.float64)
        cnt = counts_new[idx]
        n_tot = n_old + cnt
        delta = mu_new[idx] - updated.mu[idx]
        updated.mu[idx] = updated.mu[idx] \
            + delta * (cnt / n_tot)[:, None]
        updated.S[idx] = (updated.S[idx] + S_new[idx]
                          + np.einsum('ij,ij->i', delta, delta)
                          * n_old * cnt / n_tot)
        updated.n[idx] = updated.n[idx] + cnt.astype(np.int64)

    # Unassigned items consolidate among themselves into new groups
    # (set-wise; see _consolidate). On an empty tree the pool is the
    # whole wave.
    pool = np.where(~placed)[0]
    if len(pool):
        new_groups_list = _consolidate(Xn, pool, cfg)
        new_idx = updated.append_groups(Xn, new_groups_list)
        for gi, members in zip(new_idx, new_groups_list):
            assignment[members] = gi

    # Split detection on the merged tree: every group touched this
    # wave (existing groups that received items and newly formed
    # groups) is checked against the constraints. New groups satisfy
    # them by construction; the check is a safety net.
    splits = []
    touched = np.unique(assignment)
    merged_disp = updated.dispersion_rows(touched)
    for g, disp in zip(touched, merged_disp):
        if disp > cfg.max_dispersion \
                or updated.n[g] > cfg.max_children:
            splits.append(int(g))

    return WaveResult(assignment.astype(np.int64),
                      pool.astype(np.int64), splits, updated)


def place_wave(X: np.ndarray, tree: CFTable, cfg: BirchConfig) -> WaveResult:
    """Place a whole wave set-wise against the pinned tree snapshot.

    X: (items, dim) vectors; validated and unit-normalized on a copy
    (the caller's array is never mutated). The input tree is NOT
    mutated; the result carries the updated tree (joint CF merge
    applied, new groups appended). Items that win no existing group
    consolidate among themselves into new groups (see module
    docstring); `new_groups` lists every item assigned to a group
    created this wave. Groups that overflow after the merge are
    listed in splits_needed for split_group() — place_wave itself
    does not split.

    No per-item loop: scoring, acceptability and winner selection are
    matrix operations; the joint CF merge is a segment sum;
    consolidation is rounds over pinned similarity rows.

    With cfg.workers > 1, scoring runs as parallel per-block matmuls
    (exact same shortlist; see module docstring) using a thread pool
    scoped to this call. For repeated waves, BirchTree reuses one
    persistent pool instead.
    """
    Xn = _prepare_X(X, expected_dim=tree.mu.shape[1])
    return _place_wave_impl(Xn, tree, cfg)


def split_group(tree: CFTable, group: int, member_X: np.ndarray,
                cfg: BirchConfig) -> tuple:
    """Split one group into two by the classic farthest-pair rule.

    member_X: (members, dim) vectors of the group's members, in the
    caller's stable order; validated and unit-normalized on a copy.
    Seeds are the farthest pair (minimum cosine; ties -> lowest
    (i, j) lexicographically). Members assign to the nearer seed
    (ties -> first seed). If that degenerates (one side empty, e.g.
    all members identical), fall back to an index split so capacity
    splits always make progress.

    Returns (new_tree, [group, new_index], labels) where labels is a
    per-member 0/1 array (0 = stays in `group`, 1 = new group). The
    input tree is not mutated.
    """
    member_X = _prepare_X(member_X, expected_dim=tree.mu.shape[1])
    m = len(member_X)
    if m < 2:
        raise ValueError('cannot split a group with fewer than 2 members')
    if not 0 <= group < len(tree):
        raise ValueError('group index out of range')

    gram = member_X @ member_X.T
    np.fill_diagonal(gram, np.inf)          # exclude self-pairs
    flat = int(np.argmin(gram))             # first occurrence: deterministic
    seed_a, seed_b = divmod(flat, m)

    sim_a = member_X @ member_X[seed_a]
    sim_b = member_X @ member_X[seed_b]
    labels = (sim_b > sim_a).astype(np.int64)   # ties stay with seed A
    if labels.all() or not labels.any():
        labels = (np.arange(m) >= (m + 1) // 2).astype(np.int64)

    updated = tree.copy()
    parts = []
    for part in (0, 1):
        members = member_X[labels == part]
        # Two-pass stable CF recomputation from the part's members.
        mu_part = members.mean(axis=0)
        S_part = float(((members - mu_part) ** 2).sum())
        parts.append((len(members), mu_part, S_part))
    n0, mu0, S0 = parts[0]
    n1, mu1, S1 = parts[1]
    updated.n[group] = n0
    updated.mu[group] = mu0
    updated.S[group] = S0
    new_index = len(updated)
    updated.n = np.concatenate([updated.n, [n1]])
    updated.mu = np.vstack([updated.mu, mu1[None, :]])
    updated.S = np.concatenate([updated.S, [S1]])
    updated.parent = np.concatenate([updated.parent, [-1]])
    updated.ids.append(f'g{new_index}')
    return updated, [group, new_index], labels


def _check_invariants(tree: CFTable, X_all: np.ndarray, labels: np.ndarray,
                      cfg: BirchConfig, *, check_constraints: bool = True) -> None:
    """Assert the core invariants of a tree/labels pair (test helper).

    X_all: every vector the tree was built from, as stored (unit
    vectors, in arrival order). labels: group row index per item.
    Checks: group sizes sum to the item count; every item is in
    exactly one group and per-group counts match the labels; each
    group's CF (mu, S) is consistent with its members; and, when
    check_constraints, every group satisfies max_children and
    max_dispersion (1e-9 tolerance for float dust). Raises
    AssertionError describing the first violation found.
    """
    X_all = np.asarray(X_all, dtype=np.float64)
    labels = np.asarray(labels)
    assert X_all.ndim == 2, 'X_all must be 2-D'
    assert labels.ndim == 1 and len(labels) == len(X_all), \
        'labels must be 1-D with one entry per item'
    assert int(tree.n.sum()) == len(X_all), \
        f'group sizes sum to {int(tree.n.sum())}, expected {len(X_all)}'
    if len(X_all):
        assert labels.min() >= 0 and labels.max() < len(tree), \
            'label out of range'
    counts = np.bincount(labels.astype(np.int64), minlength=len(tree))
    assert (counts == tree.n).all(), \
        'per-group member counts disagree with labels'
    for g in range(len(tree)):
        members = X_all[labels == g]
        mu = members.mean(axis=0) if len(members) \
            else np.zeros(X_all.shape[1])
        assert np.allclose(tree.mu[g], mu,
                           rtol=1e-9, atol=1e-9), f'mu mismatch in group {g}'
        S = float(((members - mu) ** 2).sum()) if len(members) else 0.0
        assert abs(float(tree.S[g]) - S) <= 1e-9 * max(1.0, abs(S)), \
            f'S mismatch in group {g}'
    if check_constraints:
        assert (tree.n <= cfg.max_children).all(), \
            'max_children violated'
        disp = tree.dispersion()
        assert (disp <= cfg.max_dispersion + 1e-9).all(), \
            'max_dispersion violated'


class BirchTree:
    """Stateful wrapper: waves in, assignments out.

    Keeps every vector seen (needed to gather members for splits) and
    the assignment of each item in arrival order. partial_fit runs
    one wave: place_wave, then split_group over overflowed groups
    until none overflow. Splits only shrink groups, so resolution
    terminates (guard is scaled to the item count as a backstop).
    Stored vectors are the unit-normalized copies produced at the
    input boundary, so CF statistics and member gathering agree.
    """

    def __init__(self, cfg: BirchConfig | None = None, dim: int | None = None):
        if dim is not None and dim < 1:
            raise ValueError(f'dim must be >= 1, got {dim!r}')
        self.cfg = cfg or BirchConfig()
        self._dim = dim
        self.tree: CFTable | None = CFTable.empty(dim) if dim is not None else None
        # Contiguous assignment array (growable buffer, same pattern
        # as the vector buffer) plus a group -> member-positions
        # index, so split resolution never rebuilds or rescans the
        # full assignment. Member lists are in arrival order.
        self._labels_buf: np.ndarray | None = None
        self._members: dict = {}
        # Growable buffer holding every vector seen — the ONE retained
        # copy of the data (amortized O(n) appends; previously a full
        # np.vstack per wave, O(n^2) copying). fit() adopts its
        # normalized input matrix as this buffer (see fit).
        self._all_X_buf: np.ndarray | None = None
        self._all_X_len: int = 0
        # Persistent scoring pool for cfg.workers > 1, created lazily
        # on first use and kept for the tree's lifetime. It is
        # independent of model state (survives fit() resets); call
        # close() to release its threads explicitly.
        self._executor: ThreadPoolExecutor | None = None

    @property
    def _all_X(self) -> np.ndarray | None:
        """All vectors seen so far, arrival order (view into buffer)."""
        if self._all_X_buf is None:
            return None
        return self._all_X_buf[:self._all_X_len]

    def _append_vectors(self, Xn: np.ndarray) -> None:
        # Fast path: Xn is the next slice of the buffer itself (fit()
        # adopts the normalized input matrix as the buffer and feeds
        # waves as views into it) — just advance the length, no copy.
        buf = self._all_X_buf
        if buf is not None and len(Xn) and Xn.base is buf:
            row_off = (Xn.ctypes.data - buf.ctypes.data) \
                // (Xn.itemsize * Xn.shape[1])
            if row_off == self._all_X_len \
                    and self._all_X_len + len(Xn) <= len(buf):
                self._all_X_len += len(Xn)
                return
        need = self._all_X_len + len(Xn)
        if self._all_X_buf is None or need > len(self._all_X_buf):
            cap = max(1024, 2 * self._all_X_len, need)
            buf = np.empty((cap, Xn.shape[1]), dtype=np.float64)
            if self._all_X_len:
                buf[:self._all_X_len] = self._all_X_buf[:self._all_X_len]
            self._all_X_buf = buf
        self._all_X_buf[self._all_X_len:need] = Xn
        self._all_X_len = need

    def _append_labels(self, assignment: np.ndarray, start: int) -> None:
        """Append one wave's assignments at `start`; update the
        group -> member-positions index (arrival order preserved)."""
        need = start + len(assignment)
        if self._labels_buf is None or need > len(self._labels_buf):
            cap = max(1024, 2 * start, need)
            buf = np.empty(cap, dtype=np.int64)
            if start:
                buf[:start] = self._labels_buf[:start]
            self._labels_buf = buf
        self._labels_buf[start:need] = assignment
        members = self._members
        for i, g in enumerate(assignment):
            members.setdefault(int(g), []).append(start + i)

    @property
    def n_groups(self) -> int:
        return len(self.tree) if self.tree is not None else 0

    @property
    def labels_(self) -> np.ndarray:
        """Group row index per item, in arrival order (a copy)."""
        if self._labels_buf is None or self._all_X_len == 0:
            return np.zeros(0, dtype=np.int64)
        return self._labels_buf[:self._all_X_len].copy()

    def close(self) -> None:
        """Release the parallel-scoring thread pool (workers > 1).

        Safe to call any time (a no-op when no pool exists); the
        pool is recreated lazily on next use. Exists so long-lived
        processes that create many trees do not accumulate idle
        threads. Does not affect model state.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def _reset(self) -> None:
        self.tree = CFTable.empty(self._dim) if self._dim is not None else None
        self._labels_buf = None
        self._members = {}
        self._all_X_buf = None
        self._all_X_len = 0

    def partial_fit(self, X: np.ndarray) -> 'BirchTree':
        """Fit one wave incrementally. A 0-row input is a no-op
        (it only fixes the dimension if none is known yet)."""
        expected = self.tree.mu.shape[1] if self.tree is not None else self._dim
        Xn = _prepare_X(X, expected_dim=expected)
        if self.tree is None:
            self.tree = CFTable.empty(Xn.shape[1])
        if len(Xn) == 0:
            return self
        return self._partial_fit_prepared(Xn)

    def _partial_fit_prepared(self, Xn: np.ndarray) -> 'BirchTree':
        executor = None
        if self.cfg.workers > 1:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=int(self.cfg.workers))
            executor = self._executor
        result = _place_wave_impl(Xn, self.tree, self.cfg, executor=executor)
        self.tree = result.tree
        start = self._all_X_len
        self._append_vectors(Xn)
        self._append_labels(result.assignment, start)
        self._resolve_splits(result.splits_needed)
        return self

    def fit(self, X: np.ndarray, wave_size: int = 256) -> 'BirchTree':
        """Reset the model, then fit X in waves of wave_size.

        The whole input is validated (and normalized on a copy)
        before the reset, so invalid input leaves the previous
        model untouched. Fitting 0 rows yields an empty model.
        """
        if wave_size < 1:
            raise ValueError(f'wave_size must be >= 1, got {wave_size!r}')
        Xn = _prepare_X(X, expected_dim=self._dim)
        self._reset()
        if self.tree is None:
            self.tree = CFTable.empty(Xn.shape[1])
        # Adopt the normalized input matrix as the vector buffer:
        # waves are views into it and _append_vectors just advances
        # the length, so fit retains exactly one copy of the data.
        # (A later partial_fit grows the buffer by copying, as usual.)
        self._all_X_buf = Xn
        self._all_X_len = 0
        for start in range(0, len(Xn), wave_size):
            self._partial_fit_prepared(Xn[start:start + wave_size])
        return self

    def _resolve_splits(self, splits_needed: list) -> None:
        queue = list(splits_needed)
        total = int(self.tree.n.sum())
        limit = 100 + 4 * max(total, 1)
        guard = 0
        while queue:
            guard += 1
            if guard > limit:
                raise RuntimeError('split resolution did not converge')
            group = queue.pop(0)
            if self.tree.n[group] < 2:
                continue
            # Re-check at pop time: a queued group may already
            # satisfy the constraints; splitting it would be spurious.
            # Only this group's dispersion is needed.
            if self.tree.dispersion_rows([group])[0] \
                    <= self.cfg.max_dispersion \
                    and self.tree.n[group] <= self.cfg.max_children:
                continue
            member_positions = np.asarray(
                self._members.get(group, []), dtype=np.int64)
            if len(member_positions) < 2:
                continue
            member_X = self._all_X[member_positions]
            self.tree, (_, new_index), part = split_group(
                self.tree, group, member_X, self.cfg)
            # Remap the moved members in the label array and the
            # member index (order preserved on both sides).
            moved = member_positions[part == 1]
            kept = member_positions[part == 0]
            self._labels_buf[moved] = new_index
            self._members[group] = kept.tolist()
            self._members[new_index] = moved.tolist()
            # Re-check both parts for continued overflow (only the
            # two affected groups' dispersions are needed).
            disp = self.tree.dispersion_rows([group, new_index])
            for g, d in zip((group, new_index), disp):
                if (d > self.cfg.max_dispersion
                        or self.tree.n[g] > self.cfg.max_children) \
                        and self.tree.n[g] >= 2:
                    queue.append(g)
