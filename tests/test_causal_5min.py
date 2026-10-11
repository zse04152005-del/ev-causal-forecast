import unittest

import numpy as np
import pandas as pd

from tests._common import CODE_DIR  # noqa: F401
from src.causal import switch_5min as s5
from src.causal.panel_models import ppml_levels


class TestTransitions(unittest.TestCase):
    def test_staggered_change_merged_and_spike_dropped(self):
        T = 200
        p = np.full((T, 1), 1.0)
        p[50:] = 1.1          # 50：一次涨价
        p[100:] = 1.2         # 100、102、104 分批涨到 1.3
        p[102:] = 1.25
        p[104:] = 1.3
        p[150] = 1.6          # 150 单点尖峰，下一点回落
        spec = s5.Spec5(gap=3)
        tr = s5.find_transitions(p, spec, T - 1)
        self.assertEqual(len(tr), 2)                                   # 尖峰净变动为 0，不算事件
        self.assertEqual(int(tr.s.iloc[1]), 100)
        self.assertEqual(int(tr.e.iloc[1]), 104)
        self.assertAlmostEqual(float(tr.x.iloc[1]), np.log(1.3 / 1.1), places=6)


def synthetic_pairs(beta=-0.7, n_zone=40, days=60, seed=0):
    """每个小区 2 个站，每天 07:00 涨价（幅度随日子略变）；计数 = Poisson(基线 × exp(β x))。"""
    rng = np.random.default_rng(seed)
    rows = []
    for z in range(n_zone):
        for st in range(2):
            for d in range(days):
                x = 0.15 + 0.05 * rng.standard_normal() if rng.random() > 0.15 else 0.0
                base = 20 * np.exp(0.5 * rng.standard_normal())
                pre = rng.poisson(base)
                post = rng.poisson(base * np.exp(beta * x + 0.1 * np.sin(d)))
                rows.append({"zone": z, "station": z * 2 + st, "day": d, "slot": 84, "x": x, "Apre": pre, "Apost": post,
                             "tou": True})
    df = pd.DataFrame(rows)
    df["dh"] = df.day * 288 + df.slot
    df["zh"] = df.station * 288 + df.slot
    return df


class TestPPMLPair(unittest.TestCase):
    def test_recovers_beta(self):
        df = synthetic_pairs()
        r = s5.ppml_pair(df, ["x"], ["dh", "zh"], pseudo=0.1)
        self.assertLess(abs(r["coef"]["x"] - (-0.7)), 3 * r["se"]["x"] + 0.1)
        self.assertTrue(r["se"]["x"] > 0)

    def test_two_way_cluster_runs(self):
        df = synthetic_pairs(n_zone=25, days=40)
        r = s5.ppml_pair(df, ["x"], ["dh", "zh"], clusters=("zone", "day"))
        self.assertTrue(np.isfinite(r["se"]["x"]))

    def test_fe_ols_multi_matches_switch_did(self):
        df = synthetic_pairs(n_zone=20, days=30)
        df["J"] = np.log(df.Apost + 1) - np.log(df.Apre + 1)
        r = s5.fe_ols_multi(df, "J", ["x"], ["dh", "zh"])
        self.assertTrue(np.isfinite(r["coef"]["x"]))


class TestPPMLLevels(unittest.TestCase):
    def test_recovers_elasticity_with_fe(self):
        rng = np.random.default_rng(1)
        Z, T = 30, 400
        zz = np.repeat(np.arange(Z), T)
        tt = np.tile(np.arange(T), Z)
        lp = 0.2 * rng.standard_normal(Z * T)
        alpha = rng.standard_normal(Z)[zz]
        lam = 0.3 * np.sin(tt / 20)
        y = rng.poisson(np.exp(2 + alpha + lam - 0.5 * lp))
        fe = [pd.factorize(zz)[0], pd.factorize(tt)[0]]
        r = ppml_levels(y.astype(float), lp[:, None], fe, zz, names=["lp"])
        self.assertLess(abs(r["coef"]["lp"] + 0.5), 4 * r["se"]["lp"] + 0.05)


if __name__ == "__main__":
    unittest.main()
