"""Hand-checkable unit tests for the batched BIRCH core.

Run: python3 -m unittest test_birch  (from this directory)
"""
import unittest

import numpy as np

from batched_birch import BirchConfig, BirchTree, CFTable, place_wave, split_group
from batched_birch.core import (
    _check_invariants, _shortlist, _shortlist_blockwise)


def unit(*coords):
    v = np.array(coords, dtype=np.float64)
    return v / np.linalg.norm(v)


def at_degrees(deg):
    r = np.deg2rad(deg)
    return np.array([np.cos(r), np.sin(r)])


E0 = unit(1, 0, 0, 0)
E1 = unit(0, 1, 0, 0)
E2 = unit(0, 0, 1, 0)


def check_bt_invariants(case, bt):
    _check_invariants(bt.tree, bt._all_X, bt.labels_, bt.cfg)


class TestPlaceWave(unittest.TestCase):
    def test_empty_tree_first_wave_singletons(self):
        tree = CFTable.empty(4)
        X = np.vstack([E0, E1, E2])
        result = place_wave(X, tree, BirchConfig())
        self.assertEqual(len(result.tree), 3)
        self.assertEqual(sorted(result.new_groups.tolist()), [0, 1, 2])
        self.assertEqual(len(set(result.assignment.tolist())), 3)

    def test_input_tree_not_mutated(self):
        tree = CFTable.empty(4)
        tree.append_singletons(np.vstack([E0]))
        n_before, ls_before = tree.n.copy(), tree.mu.copy()
        place_wave(np.vstack([E0, E1]), tree, BirchConfig())
        np.testing.assert_array_equal(tree.n, n_before)
        np.testing.assert_array_equal(tree.mu, ls_before)

    def test_identical_vectors_join_existing_group(self):
        tree = CFTable.empty(4)
        tree.append_singletons(np.vstack([E0]))
        result = place_wave(np.vstack([E0, E0, E0]), tree, BirchConfig())
        self.assertEqual(result.assignment.tolist(), [0, 0, 0])
        self.assertEqual(int(result.tree.n[0]), 4)

    def test_orthogonal_vector_forms_new_group(self):
        tree = CFTable.empty(4)
        tree.append_singletons(np.vstack([E0]))
        result = place_wave(np.vstack([E1]), tree, BirchConfig())
        self.assertEqual(result.new_groups.tolist(), [0])
        self.assertEqual(len(result.tree), 2)

    def test_split_flagged_on_capacity_overflow(self):
        cfg = BirchConfig(max_children=2, max_dispersion=10.0)
        tree = CFTable.empty(4)
        tree.append_singletons(np.vstack([E0]))
        # 3 more copies join group 0 -> merged n=4 > max_children=2.
        result = place_wave(np.vstack([E0, E0, E0]), tree, cfg)
        self.assertEqual(result.splits_needed, [0])


class TestSplitGroup(unittest.TestCase):
    def test_split_preserves_members(self):
        tree = CFTable.empty(4)
        members = np.vstack([E0, E0, E1, E1])
        tree.append_singletons(np.vstack([E0]))
        tree.n[0] = 4
        # stable-CF adaptation: mu = member mean, S = two-pass
        # sum of squared deviations (was ls=sum, ss=4.0).
        tree.mu[0] = members.mean(axis=0)
        tree.S[0] = float(((members - members.mean(axis=0)) ** 2).sum())
        new_tree, indices, labels = split_group(tree, 0, members, BirchConfig())
        self.assertEqual(indices, [0, 1])
        self.assertEqual(int(new_tree.n.sum()), 4)
        # E0s together, E1s together.
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])


class TestBirchTree(unittest.TestCase):
    def test_identical_vectors_cluster_together(self):
        bt = BirchTree(BirchConfig())
        bt.partial_fit(np.vstack([E0]))
        bt.partial_fit(np.vstack([E0, E0, E0]))
        self.assertEqual(bt.n_groups, 1)
        self.assertEqual(bt.labels_.tolist(), [0, 0, 0, 0])

    def test_orthogonal_vectors_separate_groups(self):
        bt = BirchTree(BirchConfig())
        bt.partial_fit(np.vstack([E0]))
        bt.partial_fit(np.vstack([E1]))
        bt.partial_fit(np.vstack([E2]))
        self.assertEqual(bt.n_groups, 3)
        self.assertEqual(len(set(bt.labels_.tolist())), 3)

    def test_split_fires_on_overflow(self):
        cfg = BirchConfig(max_children=4, max_dispersion=10.0)
        bt = BirchTree(cfg)
        bt.partial_fit(np.vstack([E0]))
        bt.partial_fit(np.vstack([E0, E0, E0, E0]))  # merged n=5 > 4
        self.assertEqual(bt.n_groups, 2)
        self.assertEqual(int(bt.tree.n.sum()), 5)
        self.assertTrue((bt.tree.n <= 4).all())

    def test_determinism(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 16))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        a = BirchTree(BirchConfig()).fit(X, wave_size=32)
        b = BirchTree(BirchConfig()).fit(X, wave_size=32)
        np.testing.assert_array_equal(a.labels_, b.labels_)
        np.testing.assert_array_equal(a.tree.n, b.tree.n)
        np.testing.assert_array_equal(a.tree.mu, b.tree.mu)
        np.testing.assert_array_equal(a.tree.S, b.tree.S)

    def test_fit_wave_size_one_matches_partial_fit(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 8))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        a = BirchTree(BirchConfig()).fit(X, wave_size=1)
        b = BirchTree(BirchConfig())
        for row in X:
            b.partial_fit(row[None, :])
        np.testing.assert_array_equal(a.labels_, b.labels_)


class TestShortlist(unittest.TestCase):
    def test_boundary_ties_pick_lowest_indices(self):
        # Scores: indices 0..7, top-3 must be [1 (0.9), 2 (0.8), 3 (0.8)]
        # — index 5 also has 0.8 but loses the tie to lower indices.
        scores = np.array([[0.1, 0.9, 0.8, 0.8, 0.2, 0.8, 0.0, 0.3]])
        self.assertEqual(_shortlist(scores, 3).tolist(), [[1, 2, 3]])

    def test_matches_full_stable_argsort(self):
        rng = np.random.default_rng(7)
        scores = np.round(rng.normal(size=(50, 40)), 1)  # many exact ties
        expected = np.argsort(-scores, axis=1, kind='stable')[:, :5]
        np.testing.assert_array_equal(_shortlist(scores, 5), expected)


class TestConsolidation(unittest.TestCase):
    """FIX 1: within-wave consolidation of new-group items."""

    def test_first_wave_identical_vectors_consolidate(self):
        # 100 identical vectors in ONE wave on an empty tree must not
        # become 100 singletons; max_children=16 caps each group.
        X = np.tile(E0, (100, 1))
        result = place_wave(X, CFTable.empty(4), BirchConfig())
        self.assertEqual(len(result.tree), 7)  # 16*6 + 4
        self.assertEqual(sorted(result.tree.n.tolist()),
                         [4, 16, 16, 16, 16, 16, 16])
        self.assertEqual(int(result.tree.n.sum()), 100)
        self.assertEqual(result.splits_needed, [])
        _check_invariants(result.tree, X, result.assignment, BirchConfig())

    def test_first_wave_identical_via_fit(self):
        bt = BirchTree(BirchConfig()).fit(np.tile(E0, (100, 1)))
        self.assertEqual(bt.n_groups, 7)
        check_bt_invariants(self, bt)

    def test_duplicate_heavy_mixed_purity(self):
        # 50 copies each of three orthogonal vectors, interleaved.
        X = np.vstack([E0, E1, E2] * 50)
        types = np.tile([0, 1, 2], 50)
        bt = BirchTree(BirchConfig()).fit(X)
        self.assertEqual(bt.n_groups, 12)  # ceil(50/16)=4 per type
        for g in range(bt.n_groups):
            self.assertEqual(len(set(types[bt.labels_ == g].tolist())), 1)
        check_bt_invariants(self, bt)

    def test_later_wave_unassigned_consolidate(self):
        bt = BirchTree(BirchConfig())
        bt.partial_fit(np.vstack([E0]))
        bt.partial_fit(np.tile(E1, (20, 1)))  # orthogonal to E0 group
        self.assertEqual(bt.n_groups, 3)  # E0 group + 16 + 4 of E1
        e1_labels = bt.labels_[1:]
        self.assertEqual(sorted(np.bincount(e1_labels).tolist()), [0, 4, 16])
        check_bt_invariants(self, bt)

    def test_consolidation_respects_max_dispersion(self):
        # v1/v2 are each similar to v0 (cos 50° ~ 0.64) but jointly
        # too dispersed for max_dispersion=0.2: {v0,v1} disp ~0.18 ok,
        # {v0,v1,v2} disp ~0.42 not. So v2 stays out.
        cfg = BirchConfig(min_similarity=0.5, max_dispersion=0.2)
        X = np.vstack([at_degrees(0), at_degrees(50), at_degrees(-50)])
        result = place_wave(X, CFTable.empty(2), cfg)
        self.assertEqual(len(result.tree), 2)
        self.assertEqual(result.assignment[0], result.assignment[1])
        self.assertNotEqual(result.assignment[0], result.assignment[2])
        _check_invariants(result.tree, X, result.assignment, cfg)

    def test_consolidation_min_similarity_gate(self):
        v0 = at_degrees(0)
        far = np.vstack([v0, at_degrees(70)])   # cos ~0.34 < 0.5
        near = np.vstack([v0, at_degrees(40)])  # cos ~0.77 >= 0.5
        r_far = place_wave(far, CFTable.empty(2), BirchConfig())
        r_near = place_wave(near, CFTable.empty(2), BirchConfig())
        self.assertEqual(len(r_far.tree), 2)
        self.assertEqual(len(r_near.tree), 1)
        self.assertEqual(int(r_near.tree.n[0]), 2)

    def test_consolidation_respects_max_children(self):
        cfg = BirchConfig(max_children=3, max_dispersion=10.0)
        X = np.tile(E0, (10, 1))
        result = place_wave(X, CFTable.empty(4), cfg)
        self.assertEqual(sorted(result.tree.n.tolist()), [1, 3, 3, 3])
        _check_invariants(result.tree, X, result.assignment, cfg)

    def test_same_wave_items_do_not_see_each_other_existing_groups(self):
        # Pinned snapshot: A (40° from group centroid) joins group 0.
        # B (80° from centroid, but 40° from A) must NOT join group 0
        # through A — same-wave tentative assignments are invisible.
        tree = CFTable.empty(2)
        tree.append_singletons(at_degrees(0)[None, :])
        X = np.vstack([at_degrees(40), at_degrees(80)])
        result = place_wave(X, tree, BirchConfig())
        self.assertEqual(result.assignment.tolist(), [0, 1])
        self.assertEqual(len(result.tree), 2)
        self.assertEqual(int(result.tree.n[0]), 2)

    def test_wave_sizes_all_satisfy_invariants(self):
        # Wave sizes 1/32/256 may produce different partitions (no
        # subset relationship is guaranteed); all must satisfy the
        # invariants and be deterministic. wave_size=1 equals
        # per-item sequential insertion (see TestBirchTree).
        rng = np.random.default_rng(42)
        centers = rng.normal(size=(5, 16))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        X = np.vstack([c + 0.3 * rng.normal(size=(120, 16)) for c in centers])
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        for ws in (1, 32, 256):
            with self.subTest(wave_size=ws):
                a = BirchTree(BirchConfig()).fit(X, wave_size=ws)
                b = BirchTree(BirchConfig()).fit(X, wave_size=ws)
                np.testing.assert_array_equal(a.labels_, b.labels_)
                np.testing.assert_array_equal(a.tree.mu, b.tree.mu)
                check_bt_invariants(self, a)

    def test_place_wave_determinism(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 8))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        tree = CFTable.empty(8)
        tree.append_singletons(X[:3])
        r1 = place_wave(X, tree, BirchConfig())
        r2 = place_wave(X, tree, BirchConfig())
        np.testing.assert_array_equal(r1.assignment, r2.assignment)
        np.testing.assert_array_equal(r1.tree.n, r2.tree.n)
        np.testing.assert_array_equal(r1.tree.mu, r2.tree.mu)
        np.testing.assert_array_equal(r1.tree.S, r2.tree.S)
        self.assertEqual(r1.splits_needed, r2.splits_needed)


class TestConstraintEnforcement(unittest.TestCase):
    """FIX 2: post-merge constraint checks and terminating splits."""

    def test_capacity_saturation_identical_vectors(self):
        cfg = BirchConfig(max_children=4, max_dispersion=10.0)
        bt = BirchTree(cfg).fit(np.tile(E0, (10, 1)), wave_size=256)
        self.assertEqual(sorted(bt.tree.n.tolist()), [2, 4, 4])
        check_bt_invariants(self, bt)

    def test_competing_assignments_split_and_resolve(self):
        # 10 items target one group (capacity 3) in a single wave.
        cfg = BirchConfig(max_children=3, max_dispersion=10.0)
        bt = BirchTree(cfg)
        bt.partial_fit(np.vstack([E0]))
        bt.partial_fit(np.tile(E0, (10, 1)))
        self.assertEqual(sorted(bt.tree.n.tolist()), [2, 3, 3, 3])
        self.assertEqual(len(bt.labels_), 11)
        check_bt_invariants(self, bt)

    def test_recursive_capacity_splits_terminate(self):
        # 32 identical members in one group, capacity 2: index splits
        # recurse until 16 groups of 2. Identical vectors are the
        # degenerate farthest-pair case.
        cfg = BirchConfig(max_children=2, max_dispersion=10.0)
        bt = BirchTree(cfg)
        bt.partial_fit(np.vstack([E0]))
        bt.partial_fit(np.tile(E0, (31, 1)))
        self.assertEqual(bt.n_groups, 16)
        self.assertTrue((bt.tree.n == 2).all())
        check_bt_invariants(self, bt)

    def test_recursive_dispersion_split_separates_clusters(self):
        # Existing group at 0°; wave brings 5 vectors at +55° and 5
        # at -55° (each cos ~0.57 to the centroid, so all join). The
        # merged group is too dispersed (~0.63 > 0.5) and must split
        # into the two tight sides.
        bt = BirchTree(BirchConfig())
        bt.partial_fit(at_degrees(0)[None, :])
        wave = np.vstack([np.tile(at_degrees(55), (5, 1)),
                          np.tile(at_degrees(-55), (5, 1))])
        bt.partial_fit(wave)
        self.assertEqual(bt.n_groups, 2)
        a_labels = set(bt.labels_[1:6].tolist())
        b_labels = set(bt.labels_[6:].tolist())
        self.assertEqual(len(a_labels), 1)
        self.assertEqual(len(b_labels), 1)
        self.assertNotEqual(a_labels, b_labels)
        check_bt_invariants(self, bt)

    def test_splits_needed_covers_all_violations(self):
        cfg = BirchConfig(max_children=3, max_dispersion=10.0)
        tree = CFTable.empty(4)
        tree.append_singletons(np.vstack([E0]))
        result = place_wave(np.tile(E0, (10, 1)), tree, cfg)
        disp = result.tree.dispersion()
        violations = {g for g in range(len(result.tree))
                      if result.tree.n[g] > cfg.max_children
                      or disp[g] > cfg.max_dispersion}
        self.assertEqual(set(result.splits_needed), violations)
        # Structural invariants hold even before split resolution.
        _check_invariants(result.tree,
                          np.vstack([E0, np.tile(E0, (10, 1))]),
                          np.concatenate([[0], result.assignment]), cfg,
                          check_constraints=False)

    def test_invariant_helper_detects_corruption(self):
        bt = BirchTree(BirchConfig()).fit(np.vstack([E0, E0, E1]))
        bad_n = bt.tree.copy()
        bad_n.n[0] += 1
        with self.assertRaises(AssertionError):
            _check_invariants(bad_n, bt._all_X, bt.labels_, bt.cfg)
        bad_ls = bt.tree.copy()
        bad_ls.mu[0, 0] += 1.0
        with self.assertRaises(AssertionError):
            _check_invariants(bad_ls, bt._all_X, bt.labels_, bt.cfg)


class TestApiCorrectness(unittest.TestCase):
    """FIX 3: fit reset, validation, immutability, empty inputs."""

    def test_fit_resets_model(self):
        bt = BirchTree(BirchConfig())
        bt.fit(np.tile(E0, (10, 1)))
        self.assertEqual(bt.n_groups, 1)
        bt.fit(np.tile(E1, (10, 1)))
        fresh = BirchTree(BirchConfig()).fit(np.tile(E1, (10, 1)))
        self.assertEqual(len(bt.labels_), 10)
        np.testing.assert_array_equal(bt.labels_, fresh.labels_)
        np.testing.assert_array_equal(bt.tree.mu, fresh.tree.mu)
        # stable-CF adaptation: mu is the mean (was ls = 10 * E1).
        np.testing.assert_allclose(bt.tree.mu[0], E1)
        check_bt_invariants(self, bt)

    def test_repeated_fit_identical(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(120, 8))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        bt = BirchTree(BirchConfig())
        bt.fit(X, wave_size=32)
        labels1, ls1 = bt.labels_.copy(), bt.tree.mu.copy()
        bt.fit(X, wave_size=32)
        np.testing.assert_array_equal(bt.labels_, labels1)
        np.testing.assert_array_equal(bt.tree.mu, ls1)

    def test_config_validation(self):
        for kwargs in ({'min_similarity': 0}, {'min_similarity': -0.1},
                       {'min_similarity': 1.5}, {'max_dispersion': 0},
                       {'max_dispersion': -1.0}, {'max_children': 0},
                       {'candidate_count': 0}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    BirchConfig(**kwargs)
        # Valid edges must not raise.
        BirchConfig(min_similarity=1.0, max_children=1, candidate_count=1)
        BirchConfig(min_similarity=0.01, max_dispersion=1e-9)

    def test_input_validation_place_wave(self):
        tree = CFTable.empty(4)
        cfg = BirchConfig()
        with self.assertRaises(ValueError):
            place_wave(np.array([1.0, 0, 0, 0]), tree, cfg)          # 1-D
        with self.assertRaises(ValueError):
            place_wave(np.array([[np.nan, 0, 0, 0]]), tree, cfg)     # NaN
        with self.assertRaises(ValueError):
            place_wave(np.array([[np.inf, 0, 0, 0]]), tree, cfg)     # inf
        with self.assertRaises(ValueError):
            place_wave(np.zeros((1, 4)), tree, cfg)                  # zero row
        with self.assertRaises(ValueError):
            place_wave(np.vstack([unit(1, 0, 0)]), tree, cfg)        # dim 3 vs 4
        with self.assertRaises(ValueError):
            place_wave(np.zeros((2, 0)), CFTable.empty(4), cfg)      # dim 0

    def test_input_validation_partial_fit(self):
        bt = BirchTree(BirchConfig())
        bt.partial_fit(np.vstack([E0]))
        with self.assertRaises(ValueError):
            bt.partial_fit(np.vstack([unit(1, 0, 0)]))               # dim change
        with self.assertRaises(ValueError):
            bt.partial_fit(np.array([[0.0, 0, 0, 0]]))               # zero row
        with self.assertRaises(ValueError):
            BirchTree(dim=0)
        # Failed partial_fits leave the model untouched.
        self.assertEqual(bt.labels_.tolist(), [0])

    def test_non_unit_vectors_are_normalized(self):
        # Policy: normalize a copy. Scaled inputs behave exactly like
        # their unit versions; the caller's array keeps its values.
        X_scaled = np.vstack([3 * E0, 0.5 * E0, 7 * E1])
        X_before = X_scaled.copy()
        bt = BirchTree(BirchConfig()).fit(X_scaled)
        self.assertEqual(bt.labels_.tolist(), [0, 0, 1])
        # stable-CF adaptation: mu is the mean (was ls = 2 * E0).
        np.testing.assert_allclose(bt.tree.mu[0], E0)
        np.testing.assert_array_equal(X_scaled, X_before)
        tree = CFTable.empty(4)
        r_scaled = place_wave(X_scaled, tree, BirchConfig())
        r_unit = place_wave(np.vstack([E0, E0, E1]), tree, BirchConfig())
        np.testing.assert_array_equal(r_scaled.assignment, r_unit.assignment)

    def test_empty_inputs(self):
        bt = BirchTree(BirchConfig())
        bt.fit(np.zeros((0, 4)))
        self.assertEqual(bt.labels_.shape, (0,))
        self.assertEqual(bt.n_groups, 0)
        self.assertEqual(bt.tree.mu.shape, (0, 4))
        # partial_fit with 0 rows is a no-op.
        bt2 = BirchTree(BirchConfig())
        bt2.partial_fit(np.zeros((0, 4)))
        self.assertEqual(bt2.labels_.shape, (0,))
        bt2.partial_fit(np.vstack([E0]))
        bt2.partial_fit(np.zeros((0, 4)))
        self.assertEqual(bt2.labels_.tolist(), [0])
        with self.assertRaises(ValueError):
            bt2.partial_fit(np.zeros((0, 3)))  # dim mismatch still raises
        # place_wave with 0 items returns an equal (copied) tree.
        tree = CFTable.empty(4)
        tree.append_singletons(np.vstack([E0]))
        result = place_wave(np.zeros((0, 4)), tree, BirchConfig())
        self.assertEqual(result.assignment.shape, (0,))
        self.assertEqual(result.new_groups.shape, (0,))
        self.assertEqual(result.splits_needed, [])
        self.assertIsNot(result.tree, tree)
        np.testing.assert_array_equal(result.tree.mu, tree.mu)
        # fit on empty after data resets to an empty model.
        bt2.fit(np.zeros((0, 4)))
        self.assertEqual(bt2.labels_.shape, (0,))
        self.assertEqual(bt2.n_groups, 0)

    def test_input_immutability(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(60, 8))  # float64, non-unit on purpose
        X_before = X.copy()
        bt = BirchTree(BirchConfig())
        bt.fit(X, wave_size=16)
        np.testing.assert_array_equal(X, X_before)
        bt.partial_fit(X[:5])
        np.testing.assert_array_equal(X, X_before)
        # float32 input is not mutated either.
        X32 = X.astype(np.float32)
        X32_before = X32.copy()
        BirchTree(BirchConfig()).fit(X32)
        np.testing.assert_array_equal(X32, X32_before)
        # Input tree (all columns + ids) untouched by place_wave/split.
        tree = CFTable.empty(4)
        tree.append_singletons(np.vstack([E0, E1]))
        snap = (tree.n.copy(), tree.mu.copy(), tree.S.copy(),
                tree.parent.copy(), list(tree.ids))
        place_wave(np.vstack([E0, E2]), tree, BirchConfig())
        split_group(tree, 0, np.vstack([E0, E1]), BirchConfig())
        for got, want in zip((tree.n, tree.mu, tree.S, tree.parent, tree.ids), snap):
            if isinstance(want, list):
                self.assertEqual(got, want)
            else:
                np.testing.assert_array_equal(got, want)

    def test_wave_size_validation(self):
        bt = BirchTree(BirchConfig())
        with self.assertRaises(ValueError):
            bt.fit(np.vstack([E0]), wave_size=0)
        with self.assertRaises(ValueError):
            bt.fit(np.vstack([E0]), wave_size=-3)


def gen_sphere(rng, n, dim):
    X = rng.normal(size=(n, dim))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def gen_blobs(rng, n, dim):
    centers = gen_sphere(rng, 6, dim)
    per = n // 6
    X = np.vstack([c + 0.25 * rng.normal(size=(per, dim)) for c in centers])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def gen_duplicates(rng, n, dim):
    pool = gen_sphere(rng, 8, dim)
    return pool[rng.integers(0, len(pool), size=n)]


class TestRandomizedInvariants(unittest.TestCase):
    """Seeded randomized sequences: invariants after every stage."""

    def test_randomized_sequences(self):
        generators = {'sphere': gen_sphere, 'blobs': gen_blobs,
                      'duplicates': gen_duplicates}
        configs = {
            'default': BirchConfig(),
            'tight': BirchConfig(min_similarity=0.6, max_dispersion=0.15,
                                 max_children=4, candidate_count=3),
        }
        for seed in (0, 1, 2):
            for gen_name, gen in generators.items():
                for cfg_name, cfg in configs.items():
                    with self.subTest(seed=seed, gen=gen_name, cfg=cfg_name):
                        rng = np.random.default_rng(seed)
                        X = gen(rng, 300, 12)
                        bt = BirchTree(cfg)
                        # Incremental in 3 chunks, invariants each time.
                        bounds = [0, 100, 210, len(X)]
                        for lo, hi in zip(bounds, bounds[1:]):
                            bt.partial_fit(X[lo:hi])
                            check_bt_invariants(self, bt)
                        self.assertEqual(len(bt.labels_), len(X))
                        # fit() resets: same data, one shot, invariants.
                        bt.fit(X, wave_size=64)
                        self.assertEqual(len(bt.labels_), len(X))
                        check_bt_invariants(self, bt)
                        # Determinism of the whole sequence.
                        bt2 = BirchTree(cfg).fit(X, wave_size=64)
                        np.testing.assert_array_equal(bt.labels_, bt2.labels_)


class TestDispersionRows(unittest.TestCase):
    """Item 3: dispersion_rows must match dispersion() exactly."""

    def test_rows_match_full_dispersion_exactly(self):
        rng = np.random.default_rng(23)
        X = rng.normal(size=(400, 16))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        tree = CFTable.empty(16)
        tree.append_groups(X, [np.arange(i, i + 7) for i in range(0, 399, 7)])
        full = tree.dispersion()
        for rows in ([0], [3, 1, 3], list(range(len(tree))),
                     [len(tree) - 1, 0, 5]):
            got = tree.dispersion_rows(rows)
            np.testing.assert_array_equal(got, full[np.asarray(rows)])
        self.assertEqual(len(tree.dispersion_rows([])), 0)

    def test_place_wave_split_check_reads_only_touched_rows(self):
        """Per-wave split check must not compute full-tree dispersion.

        With CFTable.dispersion patched to raise, place_wave must
        still produce the identical result, and dispersion_rows must
        be asked for exactly the touched groups
        (np.unique(assignment)) — a strict subset of the tree here.
        """
        cfg = BirchConfig(min_similarity=0.5, max_dispersion=0.3,
                          max_children=4, candidate_count=3)
        rng = np.random.default_rng(41)
        X = rng.normal(size=(170, 8))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # First wave builds a multi-group tree (>= ~38 groups at
        # max_children=4); the second wave is small, so it touches
        # only a subset of the groups.
        tree = place_wave(X[:150], CFTable.empty(8), cfg).tree
        self.assertGreater(len(tree), 20)
        wave = X[150:]

        reference = place_wave(wave, tree, cfg)
        touched = np.unique(reference.assignment)
        self.assertLess(len(touched), len(reference.tree))

        calls = []
        orig_rows = CFTable.dispersion_rows
        orig_disp = CFTable.dispersion

        def spy_rows(self_, rows):
            calls.append(np.asarray(rows, dtype=np.int64))
            return orig_rows(self_, rows)

        def forbidden(self_):
            raise AssertionError('full-tree dispersion() called')

        CFTable.dispersion_rows = spy_rows
        CFTable.dispersion = forbidden
        try:
            got = place_wave(wave, tree, cfg)
        finally:
            CFTable.dispersion_rows = orig_rows
            CFTable.dispersion = orig_disp

        # Dispersion was requested for exactly the touched groups.
        self.assertTrue(calls)
        requested = np.unique(np.concatenate(calls))
        np.testing.assert_array_equal(requested, touched)
        for rows in calls:
            self.assertTrue(set(rows.tolist()) <= set(touched.tolist()))

        # And the result is identical to the unpatched run.
        np.testing.assert_array_equal(got.assignment,
                                      reference.assignment)
        np.testing.assert_array_equal(got.new_groups,
                                      reference.new_groups)
        self.assertEqual(got.splits_needed, reference.splits_needed)
        np.testing.assert_array_equal(got.tree.n, reference.tree.n)
        np.testing.assert_array_equal(got.tree.mu, reference.tree.mu)
        np.testing.assert_array_equal(got.tree.S, reference.tree.S)


class TestMemberIndex(unittest.TestCase):
    """Item 4: contiguous labels + group->member index consistency."""

    def assert_index_consistent(self, bt):
        labels = bt.labels_
        self.assertEqual(len(labels), bt._all_X_len)
        seen = []
        for g in range(len(bt.tree)):
            expected = np.where(labels == g)[0].tolist()
            self.assertEqual(bt._members.get(g, []), expected,
                             f'member index mismatch for group {g}')
            seen.extend(expected)
        self.assertEqual(sorted(seen), list(range(len(labels))))

    def test_index_consistent_after_splits(self):
        # Tight config forces frequent splits during resolution.
        cfg = BirchConfig(min_similarity=0.6, max_dispersion=0.15,
                          max_children=4, candidate_count=3)
        rng = np.random.default_rng(31)
        X = rng.normal(size=(600, 12))
        bt = BirchTree(cfg)
        for lo in range(0, len(X), 150):
            bt.partial_fit(X[lo:lo + 150])
            self.assert_index_consistent(bt)
            check_bt_invariants(self, bt)

    def test_labels_returns_a_copy(self):
        rng = np.random.default_rng(37)
        X = rng.normal(size=(100, 8))
        bt = BirchTree(BirchConfig()).fit(X, wave_size=32)
        lab = bt.labels_
        lab[:] = -999
        self.assertFalse((bt.labels_ == -999).any())

    def test_labels_empty_model(self):
        bt = BirchTree(BirchConfig(), dim=4)
        self.assertEqual(len(bt.labels_), 0)
        bt.fit(np.zeros((0, 4)))
        self.assertEqual(len(bt.labels_), 0)
        self.assertEqual(bt._members, {})


class TestRetention(unittest.TestCase):
    """Item 1: exactly one retained copy of the data matrix."""

    def test_fit_retains_single_copy(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(500, 16))
        bt = BirchTree(BirchConfig()).fit(X, wave_size=128)
        # No per-wave vector list is retained alongside the buffer.
        self.assertFalse(hasattr(bt, '_vectors'))
        # The buffer holds exactly the fit rows (no growth slack from
        # fit itself: the normalized input matrix was adopted).
        self.assertEqual(bt._all_X_len, len(X))
        self.assertEqual(len(bt._all_X_buf), len(X))
        # Retained rows are the unit-normalized input, in order.
        expected = X / np.linalg.norm(X, axis=1, keepdims=True)
        np.testing.assert_allclose(bt._all_X, expected, rtol=1e-12)
        check_bt_invariants(self, bt)

    def test_partial_fit_after_fit_grows_buffer(self):
        rng = np.random.default_rng(11)
        X = rng.normal(size=(300, 8))
        bt = BirchTree(BirchConfig()).fit(X[:200], wave_size=64)
        labels_before = bt.labels_.copy()
        bt.partial_fit(X[200:])
        self.assertEqual(bt._all_X_len, len(X))
        self.assertEqual(len(bt.labels_), len(X))
        # Earlier assignments may only change via split remaps of
        # their own group; invariants must hold regardless.
        check_bt_invariants(self, bt)
        self.assertEqual(len(labels_before), 200)

    def test_refit_replaces_retained_copy(self):
        rng = np.random.default_rng(13)
        X1 = rng.normal(size=(200, 8))
        X2 = rng.normal(size=(120, 8))
        bt = BirchTree(BirchConfig()).fit(X1, wave_size=64)
        bt.fit(X2, wave_size=64)
        self.assertEqual(bt._all_X_len, len(X2))
        self.assertEqual(len(bt.labels_), len(X2))
        check_bt_invariants(self, bt)


class TestParallelScoring(unittest.TestCase):
    """Mode B: parallel scoring blocks (workers > 1) must reproduce
    the serial shortlist and clustering exactly."""

    def test_workers_validation(self):
        for bad in (0, -1, 2.5):
            with self.subTest(workers=bad):
                with self.assertRaises(ValueError):
                    BirchConfig(workers=bad)
        BirchConfig(workers=1)
        BirchConfig(workers=8)

    def test_blockwise_shortlist_matches_serial(self):
        rng = np.random.default_rng(42)
        Xn = rng.normal(size=(30, 16))
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        C = rng.normal(size=(41, 16))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        scores = Xn @ C.T
        for k in (1, 5, 41):
            ref_idx = _shortlist(scores, k)
            ref_sc = np.take_along_axis(scores, ref_idx, axis=1)
            for w in (2, 3, 4, 7, 64):  # 64 > groups: empty blocks
                with self.subTest(k=k, workers=w):
                    idx, sc = _shortlist_blockwise(Xn, C, k, w)
                    np.testing.assert_array_equal(idx, ref_idx)
                    # Scores may differ by ~1 ULP across BLAS builds:
                    # block matmuls can use different kernels than the
                    # full matmul. Selection (indices) is exact.
                    np.testing.assert_allclose(sc, ref_sc,
                                               rtol=1e-12, atol=1e-12)

    def test_blockwise_topk_tie_across_blocks(self):
        # Groups 1 and 4 have IDENTICAL centroids and sit in
        # different blocks (workers=3 -> [0,2) [2,4) [4,6)); an item
        # aligned with them ties exactly at 1.0. The lowest group
        # index must win, exactly as in the serial shortlist.
        rng = np.random.default_rng(3)
        C = rng.normal(size=(6, 8))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        C[4] = C[1]
        Xn = C[1][None, :].copy()
        scores = Xn @ C.T
        # Identical centroids: scores tie up to ~1 ULP (BLAS kernels
        # may round the two dot products differently). The contract
        # is that blockwise selection matches the serial shortlist.
        self.assertAlmostEqual(scores[0, 1], scores[0, 4], places=12)
        for k in (1, 2):
            with self.subTest(k=k):
                idx, _ = _shortlist_blockwise(Xn, C, k, 3)
                np.testing.assert_array_equal(idx, _shortlist(scores, k))
        idx2, _ = _shortlist_blockwise(Xn, C, 2, 3)
        self.assertEqual(sorted(idx2[0].tolist()), [1, 4])

    def test_workers_equivalence_fit(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(300, 32))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        ref = BirchTree(BirchConfig(workers=1)).fit(X, wave_size=32)
        for w in (2, 4):
            with self.subTest(workers=w):
                bt = BirchTree(BirchConfig(workers=w)).fit(X, wave_size=32)
                np.testing.assert_array_equal(bt.labels_, ref.labels_)
                np.testing.assert_array_equal(bt.tree.n, ref.tree.n)
                np.testing.assert_array_equal(bt.tree.mu, ref.tree.mu)
                np.testing.assert_array_equal(bt.tree.S, ref.tree.S)
                check_bt_invariants(self, bt)
                bt.close()

    def test_workers_equivalence_place_wave(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(200, 16))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        seed_tree = BirchTree(BirchConfig()).fit(X[:120], wave_size=40).tree
        wave = X[120:]
        r1 = place_wave(wave, seed_tree, BirchConfig(workers=1))
        r4 = place_wave(wave, seed_tree, BirchConfig(workers=4))
        np.testing.assert_array_equal(r4.assignment, r1.assignment)
        np.testing.assert_array_equal(r4.new_groups, r1.new_groups)
        self.assertEqual(r4.splits_needed, r1.splits_needed)
        np.testing.assert_array_equal(r4.tree.n, r1.tree.n)
        np.testing.assert_array_equal(r4.tree.mu, r1.tree.mu)
        np.testing.assert_array_equal(r4.tree.S, r1.tree.S)

    def test_pool_lifecycle(self):
        X = np.vstack([E0, E1, E2])
        serial = BirchTree(BirchConfig(workers=1)).fit(X)
        self.assertIsNone(serial._executor)  # workers=1: no threads
        bt = BirchTree(BirchConfig(workers=2))
        self.assertIsNone(bt._executor)      # lazy: not created yet
        bt.fit(X)
        self.assertIsNotNone(bt._executor)
        bt.close()
        self.assertIsNone(bt._executor)
        bt.partial_fit(np.vstack([E0]))      # recreated on next use
        self.assertIsNotNone(bt._executor)
        bt.close()


if __name__ == '__main__':
    unittest.main()
