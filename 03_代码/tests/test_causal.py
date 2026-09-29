import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from tests._common import CODE_DIR  # noqa: F401
from src.causal.anchors import (load_anchors, merge_levels, placeholder_json, project_monotone_nonneg,
                                save_json, window_factor)
from src.causal.switch_did import PanelSpec, build_panel, estimate_cells, estimate_design, fe_ols
from src.data.prices import ring_members


def synthetic(beta=-0.6, routine_jump=0.4, days=120, seed=0):
    """分时小区在 0 点有额外的作息跳变（构成混杂），一半分时小区 0 点涨价，另一半 6 点涨价。

    设计 A（分时 vs 固定）会把 0 点的作息跳变误算成价格效应；设计 C（分时小区之间）不会。
    """
    rng = np.random.default_rng(seed)
    n_tou, n_fix = 24, 24
    N, T = n_tou + n_fix, 24 * days
    time = pd.date_range("2022-11-07", periods=T, freq="h")          # 避开节假日
    h = np.asarray(time.hour)
    lp = np.full((T, N), np.log(1.6))
    for i in range(n_tou):
        up, down = (0, 12) if i % 2 == 0 else (6, 18)
        high = ((h >= up) & (h < down)).astype(float)
        lp[:, i] += 0.10 * high - 0.05
    base = 20 + 5 * np.sin(2 * np.pi * h / 24)[:, None] * np.ones((1, N))
    routine = np.zeros((T, N))
    routine[:, :n_tou] = routine_jump * ((h >= 0) & (h < 3))[:, None]    # 分时小区 0–2 点作息更高
    dl = lp - lp.mean(axis=0, keepdims=True)
    Y = base * np.exp(routine + beta * dl) * np.exp(rng.normal(0, 0.05, (T, N)))
    pricing = np.array([2] * n_tou + [0] * n_fix)
    groups = np.array([i % 3 for i in range(N)])
    xy = rng.uniform(0, 10, (N, 2))
    dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    return Y, lp, pricing, groups, time, ring_members(dist, [[0, 2], [2, 4], [4, 6]])


class TestSwitchDID(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Y, lp, pricing, groups, time, mem = synthetic()
        cls.panel = build_panel(Y, lp, pricing, groups, time, mem, spec=PanelSpec(exclude_holiday_pm1=False))

    def test_design_c_recovers_beta(self):
        r = estimate_design(self.panel, "C")
        b, se = r["coef"]["x"], r["se"]["x"]
        self.assertLess(abs(b - (-0.6)), max(3 * se, 0.05), f"设计 C 估计 {b:.3f}（se {se:.3f}）偏离真值 −0.6")

    def test_design_a_is_confounded(self):
        a = estimate_design(self.panel, "A")["coef"]["x"]
        c = estimate_design(self.panel, "C")["coef"]["x"]
        self.assertGreater(abs(a - (-0.6)), 3 * abs(c - (-0.6)) + 0.1, f"A={a:.3f} 应明显偏离，C={c:.3f}")

    def test_deterministic_schedule_not_identified_within_zone(self):
        # 时刻表每天相同 → 小区×时刻固定效应吸收全部价格变动（命题 1 的离散版本）
        r = estimate_design(self.panel, "B")
        self.assertTrue(np.isnan(r["coef"]["x"]) or r["resid_var_share_x0"] < 1e-8)

    def test_cells_and_merge(self):
        est = estimate_cells(self.panel, 3, 4, "C")
        self.assertEqual(set(est.level), {"cell", "context", "city"})
        own = merge_levels(est, 3, 4, min_zones=1, max_se=10)
        self.assertEqual(len(own), 12)
        own_strict = merge_levels(est, 3, 4, min_zones=10 ** 6, max_se=10)
        self.assertTrue(all(e["level"] == "city" for e in own_strict))

    def test_fe_ols_simple(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"g": rng.integers(0, 20, 4000), "zone": rng.integers(0, 50, 4000)})
        df["x"] = rng.normal(size=4000) + 0.1 * df.g
        df["J"] = 2.0 * df.x + 0.5 * df.g + rng.normal(0, 0.1, 4000)
        r = fe_ols(df, "J", ["x"], ["g"])
        self.assertAlmostEqual(r["coef"]["x"], 2.0, places=2)


class TestAnchors(unittest.TestCase):
    def test_placeholder_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.json")
            save_json(placeholder_json(), p)
            a = load_anchors(p, G=3)
            self.assertTrue(a.is_placeholder)
            np.testing.assert_allclose(a.beta, -0.45)
            self.assertEqual(a.beta.shape, (3, 4))

    def test_specific_overrides_general(self):
        js = placeholder_json()
        js["source"] = "test"
        js["own"].append({"group": 1, "context": "weekday_night", "beta": -0.9, "se": 0.1})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.json")
            save_json(js, p)
            a = load_anchors(p, G=3)
        self.assertFalse(a.is_placeholder)
        self.assertAlmostEqual(a.beta[1, 1], -0.9)
        self.assertAlmostEqual(a.beta[0, 1], -0.45)

    def test_repo_anchor_files_load(self):
        for fn in ("placeholder.json", "hourly_train.json", "fivemin_train.json", "fivemin_train_pooled.json", "fivemin_main.json",
                   "fivemin_main_lag.json"):
            p = os.path.join(CODE_DIR, "configs", "anchor", fn)
            if os.path.exists(p):
                a = load_anchors(p, G=3)
                self.assertTrue(np.isfinite(a.beta).all())

    def test_lag_anchor_file(self):
        p = os.path.join(CODE_DIR, "configs", "anchor", "fivemin_main_lag.json")
        if os.path.exists(p):
            a = load_anchors(p, G=3)
            self.assertEqual(len(a.lag_weights), 3)
            self.assertAlmostEqual(float(a.lag_weights.sum()), 1.0, places=3)
            self.assertTrue(0 < a.meta["window_factor"] < 1)
            self.assertTrue((a.delta == 0).all())

    def test_window_factor_and_projection(self):
        self.assertEqual(window_factor(None), 1.0)
        self.assertAlmostEqual(window_factor([0.5, 0.5], 3), (0.5 + 1 + 1) / 3)
        np.testing.assert_allclose(project_monotone_nonneg(np.array([0.1, 0.3, -0.2])), [0.2, 0.2, 0.0])
        np.testing.assert_allclose(project_monotone_nonneg(np.array([0.3, 0.2, 0.1])), [0.3, 0.2, 0.1])


if __name__ == "__main__":
    unittest.main()
