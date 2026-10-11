import unittest

import numpy as np
import pandas as pd

from tests._common import prepared, requires_data
from src.conformal.aci import aci_calibrate
from src.eval.counterfactual import cf_error_decomposition, plugin_counterfactual
from src.eval.metrics import metrics_block, metrics_table
from src.eval.stats import dm_test, wilcoxon_zones
from src.simulate import scenarios as sc

Q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


class TestMetrics(unittest.TestCase):
    def test_perfect_and_mask(self):
        y = np.array([[0.2, 0.4]])
        yq = np.repeat(y[..., None], 7, -1)
        v = np.array([[True, False]])
        m = metrics_block(yq, y, v, Q)
        self.assertAlmostEqual(m["MAE"], 0.0)
        self.assertAlmostEqual(m["PICP90"], 1.0)
        yq2 = yq.copy()
        yq2[0, 1] += 10                                           # 被掩码的点再错也不影响
        self.assertAlmostEqual(metrics_block(yq2, y, v, Q)["MAE"], 0.0)

    def test_table_shape(self):
        rng = np.random.default_rng(0)
        yq = np.sort(rng.random((5, 24, 3, 7)), -1)
        t = metrics_table(yq, rng.random((5, 24, 3)), np.ones((5, 24, 3), bool), Q, [1, 6, 24])
        self.assertEqual(list(t.step), [1, 6, 24, "avg"])


class TestStats(unittest.TestCase):
    def test_dm(self):
        rng = np.random.default_rng(0)
        base = rng.random(300)
        r = dm_test(base * 0.8, base, h=1)
        self.assertLess(r["dm"], 0)
        self.assertLess(r["p"], 0.01)
        self.assertTrue(np.isnan(dm_test(base, base)["dm"]))

    def test_wilcoxon(self):
        a = np.linspace(1, 2, 50)
        r = wilcoxon_zones(a * 0.9, a)
        self.assertLess(r["p"], 0.01)
        self.assertEqual(r["share_better"], 1.0)


class TestACI(unittest.TestCase):
    def test_restores_coverage(self):
        rng = np.random.default_rng(0)
        n, N = 1500, 20
        y = rng.normal(0, 1, (n, N))
        lo, hi = np.full((n, N), -1.0), np.full((n, N), 1.0)      # 名义 90%，实际约 68%
        v = np.ones((n, N), bool)
        r = aci_calibrate(lo, hi, y, v, np.ones(N), alpha=0.1, gamma=0.005, window=168, h=3)
        cover = ((y >= r["lo"]) & (y <= r["hi"]))[300:].mean()
        self.assertAlmostEqual(cover, 0.9, delta=0.03)


class TestCounterfactualAndScenarios(unittest.TestCase):
    def test_plugin_ratio(self):
        yq = np.full((2, 3, 4, 7), 0.2)
        dl = np.zeros((2, 3, 4))
        ctx = np.zeros((2, 3), int)
        beta = np.full((4, 4), -0.5)
        out = plugin_counterfactual(yq, dl, dl + 0.1, beta, ctx, clip_max=None)
        np.testing.assert_allclose(out, 0.2 * np.exp(-0.05))
        d = cf_error_decomposition(np.zeros(5), np.zeros(5), -0.5, -0.8, np.full(5, 0.1))
        self.assertAlmostEqual(d["causal"], 0.03)

    def test_scenarios(self):
        t = pd.date_range("2022-11-07", periods=48, freq="h")
        p = 1.5 + 0.1 * (np.asarray(t.hour) >= 12)[:, None] * np.ones((1, 3))
        mask = np.array([True, True, False])
        np.testing.assert_allclose(sc.widen(p, t, mask, 1.0), p)
        np.testing.assert_allclose(sc.shift_schedule(p, t, mask, 24), p)
        w = sc.widen(p, t, mask, 2.0)
        self.assertAlmostEqual(float(np.ptp(w[:24, 0])), 0.2)
        s = sc.flatten_service(p, p, t, mask)
        self.assertAlmostEqual(float(np.ptp(s[:24, 0] - p[:24, 0])), 0.0)
        self.assertTrue((sc.draw_betas(np.array([-0.1]), np.array([0.5]), 100) <= 0).all())


@requires_data
class TestRealDataEval(unittest.TestCase):
    def test_switch_consistency_perfect_forecaster(self):
        from src.data.windows import WindowDataset
        from src.eval.counterfactual import switch_consistency
        P = prepared()
        W = WindowDataset(P, "test")
        f = W.origins[:, None] + np.arange(1, 25)[None]
        r = switch_consistency(P, W.origins, P.y[f])
        self.assertAlmostEqual(r["diff"], 0.0, places=8)

    def test_semisynthetic_design_d_recovers_under_endogeneity(self):
        from src.causal.switch_did import PanelSpec, build_panel, estimate_design
        from src.eval.semisynthetic import make_semisynthetic, to_prepared
        P = prepared()
        S = make_semisynthetic(P, rho=1.0, betas_by_group=(-0.8, -0.8, -0.8), flat_day_prob=0.2,
                               day_shift_prob=0.2, seed=0)
        Qp = to_prepared(P, S)
        np.testing.assert_allclose(S.true_outcome(S.lp), S.y, rtol=1e-10)
        panel = build_panel(Qp.raw["duration"], Qp.lp, Qp.pricing, Qp.groups, Qp.time, Qp.ring_members,
                            valid=~Qp.frozen, t_range=(0, Qp.split.train_end), spec=PanelSpec())
        d = estimate_design(panel, "D", exposures=False)["coef"]["x"]
        c = estimate_design(panel, "C", exposures=False)["coef"]["x"]
        self.assertLess(abs(d - (-0.8)), 0.25, f"设计 D = {d:.3f}")
        self.assertGreater(c, d, "内生时刻表下设计 C 应偏向 0 以上")


if __name__ == "__main__":
    unittest.main()
