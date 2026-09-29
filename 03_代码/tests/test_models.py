"""模型测试（需要 PyTorch；云端没有 PyTorch 时自动跳过，在 Mac / Windows 上运行）。"""
import unittest

import numpy as np

from tests._common import HAS_TORCH, DEFAULT_CFG, prepared, requires_data, requires_torch
from src.causal.anchors import AnchorSet
from src.utils.config import load_config

if HAS_TORCH:
    import torch
    from src.models.cpastgnn import CPASTGNN
    from src.models.losses import anchor_loss, hier_loss, pinball_loss

N, L, H, K, G, Q, B = 6, 8, 4, 3, 2, 7, 3


def mcfg(**kw):
    cfg = load_config(DEFAULT_CFG)
    m = cfg.model
    m.update({"D": 8, "dilations": [1, 2, 4], "head_hidden": 16, "dropout": 0.0, "clip_max": None})
    m.staeformer.update({"d_model": 8, "tod_dim": 4, "dow_dim": 4, "adaptive_dim": 8, "heads": 2, "layers": 1,
                         "ff_dim": 16})
    m.update(kw)
    return m, cfg.graph


def anchors(beta=-0.5, delta=(0.3, 0.2, 0.1), gamma=0.0):
    return AnchorSet(beta=np.full((G, 4), beta), se=np.full((G, 4), 0.2), delta=np.array(delta, float),
                     delta_se=np.full(K, 0.1), gamma=gamma, gamma_se=1.0, source="test")


def build(mode="cut", seed=0, **kw):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m, g = mcfg(**kw)
    a = rng.random((N, N))
    sup = [a / a.sum(1, keepdims=True)]
    model = CPASTGNN(m, N=N, L=L, H=H, K=K, cov_hist_dim=12, cov_fut_dim=12, static=rng.normal(size=(N, 8)),
                     supports=sup, groups=np.array([0, 0, 0, 1, 1, 1]), anchors=anchors(), anchor_mode=mode,
                     graph_cfg=g, init_quantiles=np.linspace(0.05, 0.6, Q))
    model.eval()
    return model


def batch(seed=0, lags=0):
    rng = np.random.default_rng(seed)
    b = {
        "x": rng.normal(size=(B, L, N, 2)), "cov_hist": rng.normal(size=(B, L, 12)),
        "tod_hist": rng.integers(0, 24, (B, L)), "dow_hist": rng.integers(0, 7, (B, L)),
        "dl_hist": rng.normal(0, 0.05, (B, L, N)), "cov_fut": rng.normal(size=(B, H, 12)),
        "dl_fut": rng.normal(0, 0.05, (B, H, N)), "spill_fut": rng.normal(0, 0.03, (B, H, N, K)),
        "shift_fut": rng.normal(0, 0.03, (B, H, N)), "ctx_fut": rng.integers(0, 4, (B, H)),
        "y_fut": rng.random((B, H, N)) * 0.5, "valid_fut": rng.random((B, H, N)) > 0.1,
    }
    if lags:
        b["dl_fut_ext"] = rng.normal(0, 0.05, (B, H + lags, N))
    out = {}
    for k, v in b.items():
        if k in ("tod_hist", "dow_hist", "ctx_fut"):
            out[k] = torch.as_tensor(v, dtype=torch.long)
        elif v.dtype == bool:
            out[k] = torch.as_tensor(v)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float32)
    return out


@requires_torch
class TestCPASTGNN(unittest.TestCase):
    def test_shapes_and_noncrossing(self):
        for backbone in ("graph_wavenet", "staeformer"):
            model = build(backbone=backbone)
            out = model(batch())
            self.assertEqual(tuple(out["y_q"].shape), (B, H, N, Q))
            self.assertTrue(bool((out["y_q"] > 0).all()))
            self.assertTrue(bool((out["y_q"].diff(dim=-1) >= -1e-6).all()), backbone)

    def test_clip(self):
        model = build(clip_max=0.3)
        self.assertLessEqual(float(model(batch())["y_q"].detach().max()), 0.3 + 1e-6)

    def test_cut_mode_has_no_price_parameters(self):
        model = build("cut")
        self.assertEqual(sum(p.numel() for p in model.price.parameters()), 0)
        self.assertEqual(sum(p.numel() for p in model.spill.parameters()), 0)
        bt = model.price.beta_table().numpy()
        np.testing.assert_allclose(bt, -0.5)

    def test_price_blind_baseline(self):
        model = build("cut")
        b1, b2 = batch(0), batch(0)
        b2["dl_fut"] = b2["dl_fut"] * 5 + 0.1
        b2["spill_fut"] = b2["spill_fut"] - 0.2
        b2["dl_hist"] = b2["dl_hist"] * 3
        np.testing.assert_allclose(model(b1)["b_q"].detach().numpy(), model(b2)["b_q"].detach().numpy(), rtol=1e-6)

    def test_override_identity_and_corollary1(self):
        model = build("cut")
        b = batch(1)
        base = model(b)
        same = model(b, price_override={"dl_fut": b["dl_fut"], "spill_fut": b["spill_fut"], "shift_fut": b["shift_fut"]})
        np.testing.assert_allclose(base["y_q"].detach().numpy(), same["y_q"].detach().numpy(), rtol=1e-6)
        dl2 = b["dl_fut"] + 0.1
        sp2 = b["spill_fut"] + 0.05
        cf = model(b, price_override={"dl_fut": dl2, "spill_fut": sp2})
        ratio = (cf["y_q"] / base["y_q"]).detach().numpy()
        delta = model.spill.delta().detach().numpy()
        expect = np.exp(-0.5 * 0.1 + (0.05 * delta).sum())
        np.testing.assert_allclose(ratio, expect, rtol=1e-5)          # 推论 1：比例只取决于 β、δ 与价格变化
        with self.assertRaises(KeyError):
            model(b, price_override={"x": b["x"]})

    def test_sign_and_monotone_constraints(self):
        for mode in ("soft", "free"):
            model = build(mode)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p) * 3)
            self.assertTrue(bool((model.price.beta_table() <= 0).all()), mode)
            d = model.spill.delta()
            self.assertTrue(bool((d >= 0).all()) and bool((d[:-1] >= d[1:] - 1e-7).all()), mode)
        model = build("none")
        with torch.no_grad():
            model.price.theta.fill_(0.7)
        self.assertTrue(bool((model.price.beta_table() > 0).all()))

    def test_ablation_switches(self):
        m5 = build(use_price=False)
        o = m5(batch())
        self.assertTrue(bool((o["eta"] == 0).all()))
        m6 = build(price_input=True)
        b1, b2 = batch(2), batch(2)
        b2["dl_hist"] = b2["dl_hist"] + 0.3
        self.assertFalse(np.allclose(m6(b1)["y_q"].detach().numpy(), m6(b2)["y_q"].detach().numpy()))
        m8 = build(use_shift=True)
        self.assertIsNotNone(m8.shift)

    def test_lag_kernel(self):
        model = build(price_lags=2)
        self.assertAlmostEqual(model.wf, 2 / 3, places=6)                 # 均匀权重：(1/3 + 2/3 + 1) / 3
        np.testing.assert_allclose(model.price.beta_table().numpy(), -0.5 / (2 / 3), rtol=1e-6)  # 长期弹性
        with torch.no_grad():
            model.price.lag_w.copy_(torch.tensor([1.0, 0.0, 0.0]))
        b = batch(3, lags=2)
        b["dl_fut_ext"][:, 2:] = b["dl_fut"]
        eta_lag = model.price(b["dl_fut_ext"], b["ctx_fut"])
        expect = model.price.beta_table().T[b["ctx_fut"]] * b["dl_fut"]
        np.testing.assert_allclose(eta_lag.detach().numpy(), expect.detach().numpy(), rtol=1e-6)

    def test_override_with_lags_keeps_past_prices(self):
        model = build(price_lags=2)                                      # 均匀权重，长期 β = −0.75
        b = batch(5, lags=2)
        b["dl_fut_ext"][:, 2:] = b["dl_fut"]
        base = model(b)["y_q"]
        cf = model(b, price_override={"dl_fut": b["dl_fut"] + 0.1})["y_q"]
        r = (cf / base).detach().numpy()
        np.testing.assert_allclose(r[:, 0], np.exp(-0.75 * 0.1 / 3), rtol=1e-5)   # 第 1 步只有 1/3 的权重在未来
        np.testing.assert_allclose(r[:, 1], np.exp(-0.75 * 0.2 / 3), rtol=1e-5)
        np.testing.assert_allclose(r[:, 2:], np.exp(-0.75 * 0.1), rtol=1e-5)

    def test_losses_and_training_step(self):
        model = build("soft")
        model.train()
        b = batch(4)
        q = model.quantiles
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        first = None
        for _ in range(80):
            out = model(b)
            loss = pinball_loss(out["y_q"], b["y_fut"], b["valid_fut"], q)
            if first is None:
                first = float(loss)
            total = loss + anchor_loss(model, anchors(), np.array([1, 1, 0, 1, 0, 0], bool),
                                       np.array([0, 0, 0, 1, 1, 1])) + 0.1 * hier_loss(model)
            opt.zero_grad()
            total.backward()
            opt.step()
        self.assertLess(float(loss), 0.7 * first)
        self.assertIsNotNone(model.price.theta.grad)

    def test_pinball_mask(self):
        yq = torch.zeros(1, 1, 2, 1)
        y = torch.tensor([[[1.0, float("nan")]]])                         # 掩码点的非有限值不影响损失
        v = torch.tensor([[[True, False]]])
        self.assertAlmostEqual(float(pinball_loss(yq, y, v, [0.5])), 0.5)


@requires_torch
@requires_data
class TestRealBatch(unittest.TestCase):
    def test_forward_on_real_batch(self):
        from src.causal.anchors import load_anchors
        from src.data.windows import WindowDataset
        from src.utils.paths import project_root, resolve
        cfg = load_config(DEFAULT_CFG, ["data.zones=20", "model.D=8"])
        P = prepared(["data.zones=20"])
        W = WindowDataset(P, "train")
        a = load_anchors(resolve(cfg.anchor.file, project_root(cfg.paths.get("root"))), G=P.G, K=P.K)
        model = CPASTGNN(cfg.model, N=P.N, L=24, H=24, K=P.K, cov_hist_dim=12, cov_fut_dim=W.n_cov_fut,
                         static=P.static, supports=P.supports, groups=P.groups, anchors=a, graph_cfg=cfg.graph,
                         init_quantiles=np.linspace(0.02, 0.5, 7))
        b = W.batch([0, 1])
        tb = {k: torch.as_tensor(v, dtype=torch.long if k in ("t0", "ctx_fut", "tod_hist", "dow_hist") else
                                 (torch.bool if v.dtype == bool else torch.float32)) for k, v in b.items()}
        out = model(tb)
        self.assertEqual(tuple(out["y_q"].shape), (2, 24, 20, 7))
        self.assertTrue(bool(torch.isfinite(out["y_q"]).all()))


if __name__ == "__main__":
    unittest.main()
