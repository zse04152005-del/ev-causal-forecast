import os
import unittest

import numpy as np
import pandas as pd

from tests._common import data_dir, prepared, requires_data  # noqa: F401
from src.data import calendar as cal
from src.data.prices import classify_pricing, ring_exposure, ring_members, shift_feature
from src.data.quality import frozen_mask, stale_flag
from src.data.scaling import Standardizer
from src.data.splits import make_split


class TestQuality(unittest.TestCase):
    def test_frozen_rules(self):
        T = 60
        occ = np.arange(T, dtype=float)[:, None].repeat(4, axis=1)   # 默认每小时都变
        dur = np.arange(T, dtype=float)[:, None].repeat(4, axis=1) + 0.5
        occ[10:16, 0], dur[10:16, 0] = 3, 2.5        # 6 小时非整数 → 冻结
        occ[10:15, 1], dur[10:15, 1] = 3, 2.5        # 5 小时 → 不算
        occ[10:33, 2], dur[10:33, 2] = 3, 2.0        # 23 小时整数 → 不算
        occ[10:34, 3], dur[10:34, 3] = 3, 2.0        # 24 小时整数 → 冻结
        m = frozen_mask(occ, dur)
        self.assertTrue(m[10:16, 0].all() and not m[16:, 0].any() and not m[:10, 0].any())
        self.assertFalse(m[:, 1].any())
        self.assertFalse(m[:, 2].any())
        self.assertTrue(m[10:34, 3].all())

    def test_stale_flag_is_causal(self):
        rng = np.random.default_rng(0)
        occ, dur = rng.integers(0, 5, (50, 3)).astype(float), rng.random((50, 3))
        occ[20:40, 1], dur[20:40, 1] = 2, 0.3
        a = stale_flag(occ, dur, 6)
        occ2, dur2 = occ.copy(), dur.copy()
        occ2[30:], dur2[30:] = 9, 0.9                # 改变未来
        b = stale_flag(occ2, dur2, 6)
        np.testing.assert_array_equal(a[:30], b[:30])
        self.assertEqual(a[24, 1], 0.0)
        self.assertEqual(a[25, 1], 1.0)


class TestCalendar(unittest.TestCase):
    def test_holidays_and_makeup(self):
        t = pd.DatetimeIndex(["2022-10-03 10:00", "2022-10-08 10:00", "2022-10-08 22:00", "2022-09-11 22:00",
                              "2022-11-16 10:00", "2023-01-29 03:00"])
        np.testing.assert_array_equal(cal.is_restday(t), [True, False, False, True, False, False])
        np.testing.assert_array_equal(cal.context_index(t), [2, 0, 1, 3, 0, 1])
        self.assertEqual(cal.calendar_features(t).shape, (6, 6))


class TestPrices(unittest.TestCase):
    def test_classify(self):
        t = pd.date_range("2022-09-01", periods=240, freq="h")
        p = np.full((240, 3), 1.5)
        p[:, 0] += 0.1 * (t.hour >= 12)               # 每天都变 → TOU
        p[[5, 6], 1] += 0.01                          # 只有 1 天变 → weak
        share, label = classify_pricing(p, t)
        self.assertEqual(list(label), ["TOU", "weak", "fixed"])
        self.assertAlmostEqual(share[1], 0.1)

    def test_ring_exposure(self):
        d = np.array([[0, 1.0, 3.0], [1.0, 0, 5.0], [3.0, 5.0, 0]])
        m = ring_members(d, [[0, 2], [2, 4]])
        dl = np.array([[0.1, 0.2, 0.3]])
        s = ring_exposure(dl, m)
        np.testing.assert_allclose(s[0, 0], [0.2, 0.3])
        np.testing.assert_allclose(s[0, 1], [0.1, 0.0])
        np.testing.assert_allclose(s[0, 2], [0.0, 0.1])
        np.testing.assert_allclose(shift_feature(np.array([[0.0], [0.1], [0.3]]))[:, 0], [0.1, 0.2, 0.0])


class TestSplitScaler(unittest.TestCase):
    def test_split_dates(self):
        t = pd.date_range("2022-09-01", "2023-02-28 23:00", freq="h")
        s = make_split(t)
        self.assertEqual(s.describe(t), {"train": ("2022-09-01", "2023-01-05"), "val": ("2023-01-06", "2023-01-23"),
                                         "test": ("2023-01-24", "2023-02-28")})

    def test_scaler_ignores_invalid(self):
        x = np.array([[1.0, 2.0], [3.0, 100.0]])
        v = np.array([[True, True], [True, False]])
        s = Standardizer().fit(x, v)
        self.assertAlmostEqual(float(s.mean), 2.0)


@requires_data
class TestRealData(unittest.TestCase):
    def test_counts_match_audit(self):
        P = prepared()
        self.assertEqual((P.T, P.N), (4344, 275))
        self.assertEqual((P.meta["n_tou"], P.meta["n_weak"], P.meta["n_fixed"]), (90, 27, 158))
        self.assertAlmostEqual(P.meta["frozen_share"], 0.0907, places=3)
        self.assertLessEqual(float(P.y.max()), 1.0 + 1e-6)          # 利用率不超过 1
        self.assertEqual([int((P.groups == g).sum()) for g in range(3)], [36, 99, 140])

    def test_frozen_matches_audit_file(self):
        path = os.path.join(os.path.dirname(os.path.dirname(data_dir())), "..", "processed", "audit", "frozen_mask.csv")
        path = os.path.normpath(path)
        if not os.path.exists(path):
            self.skipTest("没有数据核查的冻结掩码文件")
        ref = pd.read_csv(path, index_col=0).to_numpy().astype(bool)
        np.testing.assert_array_equal(prepared().frozen, ref)

    def test_windows_no_leakage(self):
        from src.data.windows import WindowDataset
        P = prepared()
        for part in ("train", "val", "test"):
            W = WindowDataset(P, part)
            lo, hi = P.split.part_range(part)
            self.assertGreaterEqual(W.origins.min() + 1, lo)
            self.assertLessEqual(W.origins.max() + W.H, hi)
            self.assertGreaterEqual(W.origins.min() - W.L + 1, 0)
        W = WindowDataset(P, "val")
        b = W.batch([0])
        np.testing.assert_array_equal(b["y_fut"][0], P.y[W.origins[0] + 1: W.origins[0] + 25])

    def test_reference_price_uses_train_only(self):
        P = prepared()
        np.testing.assert_allclose(P.ref_lp, P.lp[: P.split.train_end + 1].mean(axis=0))
        fixed = P.pricing == 0
        self.assertLess(float(np.abs(P.dl[:, fixed]).max()), 1e-6)   # 固定电价小区 Δℓ 恒为 0


if __name__ == "__main__":
    unittest.main()
