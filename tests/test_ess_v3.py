"""半合成基准 v3 的新选项：负的 ρ、同质真实弹性、内生偏离、事后矩与 T3 区间（不依赖 PyTorch）。"""
import os
import unittest

import numpy as np

from tests._common import prepared, requires_data, requires_torch


@requires_data
class TestESSv3(unittest.TestCase):
    def setUp(self):
        self.P = prepared()

    def _gen(self, **kw):
        from src.eval.semisynthetic import make_semisynthetic
        return make_semisynthetic(self.P, seed=0, **kw)

    def test_negative_rho_puts_high_prices_in_low_demand_hours(self):
        S_pos, S_neg = self._gen(rho=1.0), self._gen(rho=-1.0)
        tr = self.P.split.train_end
        for S, sign in ((S_pos, 1), (S_neg, -1)):
            j = np.flatnonzero(S.treated)[0]
            prof = np.array([S.y_base[: tr + 1][self.P.tod[: tr + 1] == h, j].mean() for h in range(24)])
            dl = np.array([(S.lp - S.ref_lp)[: tr + 1][self.P.tod[: tr + 1] == h, j].mean() for h in range(24)])
            self.assertEqual(np.sign(np.corrcoef(prof, dl)[0, 1]), sign)

    def test_homogeneous_beta(self):
        S = self._gen(rho=0.0, beta_true=-0.2)
        self.assertTrue(np.allclose(S.beta, -0.2))

    def test_endogenous_shifts_follow_demand(self):
        S0 = self._gen(rho=0.0, day_shift_prob=1.0, endo=0.0)
        S1 = self._gen(rho=0.0, day_shift_prob=1.0, endo=1.0)
        self.assertFalse(np.allclose(S0.lp, S1.lp))

    def test_rho_out_of_range(self):
        with self.assertRaises(ValueError):
            self._gen(rho=1.5)

    def test_report_moments_and_T3(self):
        from types import SimpleNamespace
        from src.eval.ess import ESSRun, evaluate_ess, scenarios
        from src.eval.semisynthetic import to_prepared
        S = self._gen(rho=0.5, beta_true=-0.5, day_shift_prob=0.2)
        Q = to_prepared(self.P, S)
        E = ESSRun(S=S, Q=Q, anchor_path="", scenarios=scenarios(S, ("flat",)))
        H, qs = 24, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        origins = np.arange(Q.split.val_end + 1, Q.T - H - 1, 24)
        fut = origins[:, None] + np.arange(1, H + 1)[None, :]
        y = S.y[fut]
        yq = np.stack([y * np.exp(z * 0.3) for z in (-1.645, -1.28, -0.67, 0, 0.67, 1.28, 1.645)], -1)  # 以真值为中心的分位数
        G, C = int(Q.groups.max()) + 1, 4
        rep = evaluate_ess(E, origins, H, yq, {"flat": None}, qs, None, np.full((G, C), -0.5),
                           clip=1.0, anchor_se_gc=np.full((G, C), 0.05))
        sc = rep["scenarios"]["flat"]
        m = sc["moments"]
        mse_true = (0.25 * m["Sdd"] - 2 * (-0.5) * m["Sdr"] + m["Srr"]) / m["n"]
        self.assertLess(mse_true, 1e-3)                                   # 真值弹性的乘性效应 ≈ 真实效应（只差截断）
        self.assertGreater(sc["T3_interval"]["coverage_cf"], 0.85)
        self.assertGreaterEqual(sc["T3_interval"]["coverage_cf"] + 1e-9, sc["T3_interval"]["coverage_fact"] - 0.05)


@requires_torch
class TestProjectedMode(unittest.TestCase):
    def test_projection_keeps_beta_nonpositive_and_gradient_flows(self):
        import torch
        from src.causal.anchors import AnchorSet
        from src.models.heads.price import PriceResponse
        G, C, N = 2, 4, 5
        anc = AnchorSet(beta=np.full((G, C), -0.2), se=np.full((G, C), 0.1), delta=np.zeros(3), delta_se=np.ones(3))
        pr = PriceResponse(np.array([0, 0, 1, 1, 1]), anc, mode="proj", init_beta=0.0)
        dl = torch.full((2, 3, N), 0.1)
        ctx = torch.zeros((2, 3), dtype=torch.long)
        loss = (pr(dl, ctx) - 0.05).pow(2).sum()                       # pushes beta upwards (towards +0.5)
        loss.backward()
        self.assertGreater(float(pr.theta.grad.abs().sum()), 0.0)      # no vanishing gradient at beta = 0
        with torch.no_grad():
            pr.theta += 1.0
        pr.project_()
        self.assertTrue(bool((pr.beta_table() <= 1e-7).all()))


if __name__ == "__main__":
    unittest.main()
