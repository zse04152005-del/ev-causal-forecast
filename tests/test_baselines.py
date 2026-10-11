"""深度基线测试。numpy 部分（图矩阵、点预测转分位数、价格响应读出）云端可跑；模型部分需要 PyTorch。"""
import types
import unittest

import numpy as np

from tests._common import DEFAULT_CFG, HAS_TORCH, prepared, requires_data, requires_torch
from src.data.graphs import cheb_polynomials, scaled_laplacian, sym_norm_adj
from src.eval.price_readout import arc_elasticity
from src.models.point_intervals import point_to_quantiles, residual_quantiles
from src.models.pseudo_samples import draw_price_shocks, mix_rows, pseudo_targets
from src.utils.config import load_config

if HAS_TORCH:
    import torch
    from src.models.baselines import NAMES, PAG, PIAST, build_baseline, readout_keys_of
    from src.models.baselines.pag import fomaml_pretrain, pseudo_batch

N, L, H, Q, B = 6, 24, 4, 7, 3
ENCODERS = ("fcnn", "lstm", "gcn", "gcnlstm", "stgcn", "astgcn", "agcrn")


def ring_adj(n=N):
    a = np.zeros((n, n), dtype=np.float32)
    for i in range(n - 1):                                    # 链 0-1-2-…，最后一个节点孤立
        if i < n - 2:
            a[i, i + 1] = a[i + 1, i] = 1
    return a


class TestGraphMatrices(unittest.TestCase):
    def test_sym_norm_and_isolated(self):
        a = ring_adj()
        s = sym_norm_adj(a)
        np.testing.assert_allclose(s, s.T, atol=1e-7)
        self.assertAlmostEqual(float(s[-1, -1]), 1.0, places=6)            # 孤立节点只剩自环
        lt = scaled_laplacian(a)
        ev = np.linalg.eigvalsh(lt.astype(np.float64))
        self.assertGreaterEqual(ev.min(), -1 - 1e-5)
        self.assertLessEqual(ev.max(), 1 + 1e-5)
        c = cheb_polynomials(lt, 3)
        self.assertEqual(c.shape, (3, N, N))
        np.testing.assert_allclose(c[0], np.eye(N), atol=1e-7)
        np.testing.assert_allclose(c[2], 2 * lt @ lt - np.eye(N), atol=1e-5)


class TestPAGPseudoSamples(unittest.TestCase):
    def test_formula_8_and_9(self):
        a = ring_adj()                                                      # 链 0-1-2-3-4，节点 5 孤立
        y = np.full((2, N), 0.4)
        shocks = np.zeros(N)
        shocks[2] = 0.1                                                     # 节点 2 涨价 10%
        out = pseudo_targets(y, shocks, -1.48, a)
        dy_own = -1.48 * 0.1 * 0.4
        self.assertAlmostEqual(out[0, 2] - 0.4, dy_own, places=9)           # 公式 (8)：本区需求下降
        self.assertAlmostEqual(out[0, 1] - 0.4, -dy_own / 2, places=9)      # 公式 (9)：两个邻区各得一半
        self.assertAlmostEqual(out[0, 3] - 0.4, -dy_own / 2, places=9)
        self.assertAlmostEqual(out[0, 0], 0.4, places=9)
        self.assertAlmostEqual(out.sum() - y.sum(), 0.0, places=9)          # 需求只在区间转移，总量不变
        big = pseudo_targets(y, np.full(N, -0.9), -3.0, a, clip_max=1.0)
        self.assertTrue((big >= 0).all() and (big <= 1).all())

    def test_shocks_and_mixing(self):
        rng = np.random.default_rng(0)
        s = np.stack([draw_price_shocks(1000, rng) for _ in range(5)])
        self.assertAlmostEqual(float((s != 0).mean()), 0.4, delta=0.03)
        self.assertGreaterEqual(float(s.min()), -0.9)
        np.testing.assert_array_equal(mix_rows(10, 0.3), [False] * 7 + [True] * 3)
        self.assertFalse(mix_rows(10, 0.0).any())
        self.assertTrue(mix_rows(10, 1.0).all())


class TestPointIntervalsAndReadout(unittest.TestCase):
    def test_residual_quantiles(self):
        rng = np.random.default_rng(0)
        y = rng.random((500, 2, 3))
        pt = y - rng.normal(0, 0.1, y.shape)
        v = np.ones_like(y, dtype=bool)
        v[:, 1] = False                                                     # 第 2 步没有有效点 → 残差分位数取 0
        rq = residual_quantiles(pt, y, v, [0.1, 0.5, 0.9])
        self.assertEqual(rq.shape, (2, 3))
        self.assertAlmostEqual(rq[0, 1], 0.0, delta=0.02)
        self.assertAlmostEqual(rq[0, 2] - rq[0, 0], 2 * 1.2816 * 0.1, delta=0.03)
        np.testing.assert_array_equal(rq[1], 0.0)
        yq = point_to_quantiles(pt, rq, clip_max=1.0)
        self.assertEqual(yq.shape, (500, 2, 3, 3))
        self.assertTrue((np.diff(yq, axis=-1) >= 0).all() and yq.min() >= 0 and yq.max() <= 1)

    def test_arc_elasticity(self):
        rng = np.random.default_rng(1)
        base = rng.random((10, 4, 5)) + 0.1
        up = base * np.exp(-0.8 * np.log1p(0.05))                         # 真实弹性 −0.8
        r = arc_elasticity(base, up, 0.05, np.array([1, 1, 0, 1, 0], bool))
        self.assertAlmostEqual(r["mean"], -0.8, places=6)
        self.assertEqual(r["n_zones"], 3)
        self.assertEqual(r["share_negative"], 1.0)
        self.assertAlmostEqual(arc_elasticity(base, base, 0.05)["mean"], 0.0, places=9)


def cfg_for(name, **over):
    ov = [f"baseline.name={name}", "baseline.D=8", "model.head_hidden=16", "baseline.dropout=0.0",
          "baseline.stgcn.blocks=[[8,4,8],[8,4,8]]", "baseline.astgcn.chev_filter=8", "baseline.astgcn.time_filter=8",
          "baseline.agcrn.hidden=8", "baseline.piast.mlp_hidden=16", "baseline.piast.dropout=0.0",
          "baseline.pag.dropout=0.0"]
    ov += [f"{k}={v}" for k, v in over.items()]
    cfg = load_config(DEFAULT_CFG, ov)
    cfg.data["H"] = H
    return cfg


def fake_prepared():
    return types.SimpleNamespace(N=N, adj=ring_adj(), ref_lp=np.log(np.full(N, 1.0, dtype=np.float32)))


def batch(seed=0):
    rng = np.random.default_rng(seed)
    b = {
        "x": rng.normal(size=(B, L, N, 2)), "cov_hist": rng.normal(size=(B, L, 12)),
        "dl_hist": rng.normal(0, 0.05, (B, L, N)), "cov_fut": rng.normal(size=(B, H, 12)),
        "dl_fut": rng.normal(0, 0.05, (B, H, N)), "y_fut": rng.random((B, H, N)) * 0.5,
        "valid_fut": rng.random((B, H, N)) > 0.1,
    }
    return {k: (torch.as_tensor(v) if v.dtype == bool else torch.as_tensor(v, dtype=torch.float32)) for k, v in b.items()}


def build(name, seed=0, **over):
    torch.manual_seed(seed)
    m = build_baseline(cfg_for(name, **over), fake_prepared(), 12, np.linspace(0.02, 0.5, Q), seed=seed)
    m.eval()
    return m


@requires_torch
class TestEncoders(unittest.TestCase):
    def test_shapes_noncrossing_all(self):
        for name in ENCODERS:
            for pi in ("none", "hist", "hist_fut"):
                out = build(name, **{"baseline.price_input": pi})(batch())
                yq = out["y_q"]
                self.assertEqual(tuple(yq.shape), (B, H, N, Q), (name, pi))
                self.assertTrue(bool(torch.isfinite(yq).all()), (name, pi))
                self.assertTrue(bool((yq >= 0).all()) and bool((yq.diff(dim=-1) >= -1e-6).all()), (name, pi))
                self.assertLessEqual(float(yq.detach().max()), 1.0 + 1e-6)

    def test_price_blind_and_price_aware(self):
        for name in ENCODERS:
            m = build(name)
            b1, b2 = batch(0), batch(0)
            b2["dl_fut"] = b2["dl_fut"] + 0.2
            b2["dl_hist"] = b2["dl_hist"] * 3
            np.testing.assert_allclose(m(b1)["y_q"].detach().numpy(), m(b2)["y_q"].detach().numpy(), rtol=1e-6)
            self.assertEqual(readout_keys_of(m), ())
            mh = build(name, **{"baseline.price_input": "hist"})
            self.assertEqual(readout_keys_of(mh), ("dl_hist",))
            b4 = batch(0)
            self.assertFalse(np.allclose(mh(b4)["y_q"].detach().numpy(),
                                         mh(b4, price_override={"dl_hist": b4["dl_hist"] + 0.2})["y_q"].detach().numpy()), name)
            mf = build(name, **{"baseline.price_input": "hist_fut"})
            self.assertEqual(readout_keys_of(mf), ("dl_hist", "dl_fut"))
            b3 = batch(0)
            base = mf(b3)["y_q"].detach().numpy()
            cf = mf(b3, price_override={"dl_fut": b3["dl_fut"] + 0.2})["y_q"].detach().numpy()
            self.assertFalse(np.allclose(base, cf), name)                   # 未来价格确实进入预测
            with self.assertRaises(KeyError):
                mf(b3, price_override={"x": b3["x"]})

    def test_point_loss_and_training_step(self):
        m = build("fcnn", **{"baseline.loss": "mse"})
        self.assertEqual(tuple(m(batch())["y"].shape), (B, H, N))
        from src.models.losses import pinball_loss
        for name in ("stgcn", "agcrn"):
            m = build(name)
            m.train()
            b = batch(4)
            opt = torch.optim.Adam(m.parameters(), lr=1e-2)
            first = None
            for _ in range(60):
                loss = pinball_loss(m(b)["y_q"], b["y_fut"], b["valid_fut"], m.quantiles)
                first = loss.item() if first is None else first
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.assertLess(loss.item(), 0.8 * first, name)

    def test_short_input_rejected(self):
        with self.assertRaises(ValueError):
            build("stgcn", **{"data.L": 8})


@requires_torch
class TestPIAST(unittest.TestCase):
    def test_forward_physics_and_clamp(self):
        m = build("piast")
        self.assertIsInstance(m, PIAST)
        self.assertEqual(readout_keys_of(m), ("dl_fut",))
        b = batch(0)
        y = m(b)["y"]
        self.assertEqual(tuple(y.shape), (B, H, N))
        m.train()
        y2, con1 = m.physics(b)
        self.assertEqual(tuple(con1.shape), (B, H, N))
        loss = ((y2 - b["y_fut"]) ** 2).mean() + (con1 ** 2).mean() + ((m.lambda_1 + 0.45) ** 2).mean()
        loss.backward()
        self.assertIsNotNone(m.lambda_1.grad)
        self.assertIsNotNone(m.net.gat.W.grad)                               # 修正后：GAT 头参与训练
        with torch.no_grad():
            m.lambda_1.fill_(-2.0)
        m.clamp_lambda(-1.0, 0.0)
        np.testing.assert_allclose(m.lambda_1.detach().numpy(), -1.0)
        m.clamp_lambda(None, None)                                           # 不截断时不改变

    def test_attention_normalization_and_price_response(self):
        m = build("piast")
        fea = torch.randn(2, N, m.net.seq)
        att = m.net.gat(fea)
        np.testing.assert_allclose(att.sum(dim=1).detach().numpy(), 1.0, rtol=1e-5)   # 原代码在第 0 维 softmax
        self.assertLess(float(att[0, 0, N - 1]), 1e-6)                       # 非邻接位置被屏蔽
        b = batch(1)
        base = m(b)["y"].detach().numpy()
        cf = m(b, price_override={"dl_fut": b["dl_fut"] + 0.1})["y"].detach().numpy()
        self.assertFalse(np.allclose(base, cf))

    def test_released_code_quirks(self):
        m = build("piast", **{"baseline.piast.released_code_quirks": "true"})
        names = {n for n, _ in m.named_parameters()}
        self.assertNotIn("net.gat.W", names)                                 # 发布代码：GAT 头不训练
        fea = torch.randn(3, N, m.net.seq)
        att = m.net.gat(fea)
        np.testing.assert_allclose(att[0].detach().numpy(), att[2].detach().numpy())   # 全批共用第 1 个样本的注意力
        m.train()
        _, con1 = m.physics(batch(2))
        self.assertTrue(con1.requires_grad)

    def test_single_step_has_no_horizon_embedding(self):
        cfg = cfg_for("piast")
        cfg.data["H"] = 1
        m = build_baseline(cfg, fake_prepared(), 12, None)
        self.assertIsNone(m.net.hemb)


@requires_torch
class TestPAG(unittest.TestCase):
    def test_forward_and_price_history(self):
        m = build("pag")
        self.assertIsInstance(m, PAG)
        self.assertEqual(readout_keys_of(m), ("dl_hist",))
        b = batch(0)
        y = m(b)["y"]
        self.assertEqual(tuple(y.shape), (B, H, N))
        same = m(b, price_override={"dl_fut": b["dl_fut"] + 0.3})["y"]            # 不看未来价格
        np.testing.assert_allclose(y.detach().numpy(), same.detach().numpy(), rtol=1e-6)
        diff = m(b, price_override={"dl_hist": b["dl_hist"] + 0.3})["y"]
        self.assertFalse(np.allclose(y.detach().numpy(), diff.detach().numpy()))

    def test_pseudo_batch_and_pretrain_moves_parameters(self):
        m = build("pag")
        b = batch(1)
        rng = np.random.default_rng(0)
        pb = pseudo_batch(b, -1.48, 0.5, rng, ring_adj(), 1.0)
        rows = mix_rows(B, 0.5)
        np.testing.assert_allclose(pb["dl_hist"][~torch.as_tensor(rows)].numpy(), b["dl_hist"][~torch.as_tensor(rows)].numpy())
        self.assertFalse(torch.equal(pb["y_fut"], b["y_fut"]))
        self.assertIs(pseudo_batch(b, -1.48, 0.0, rng, ring_adj(), 1.0), b)

        class _W:                                                               # 最小窗口数据集替身
            def __init__(self, bt):
                self.bt = {k: v.numpy() for k, v in bt.items()}

            def iterate(self, bs, shuffle=False, rng=None, max_batches=None):
                yield self.bt

        before = {k: v.detach().clone() for k, v in m.state_dict().items()}
        hist = fomaml_pretrain(m, _W(b), _W(batch(2)), [-1.48, -0.228], 2, torch.device("cpu"), rng, ring_adj(),
                               1.0, B)
        self.assertEqual(len(hist), 2)
        self.assertTrue(all(np.isfinite(hist)))
        moved = any(not torch.equal(before[k], v) for k, v in m.state_dict().items())
        self.assertTrue(moved)                                                  # 与发布代码不同：全局参数确实被更新


@requires_torch
@requires_data
class TestBaselinesRealBatch(unittest.TestCase):
    def test_forward_on_real_batch(self):
        from src.data.windows import WindowDataset
        from src.models.torch_utils import to_tensors
        P = prepared(["data.zones=20"])
        W = WindowDataset(P, "train")
        tb = to_tensors(W.batch([0, 1]), torch.device("cpu"))
        for name in ("gcn", "stgcn", "astgcn", "agcrn", "pag", "piast"):
            cfg = load_config(DEFAULT_CFG, [f"baseline.name={name}", "baseline.D=8", "baseline.agcrn.hidden=8",
                                            "baseline.stgcn.blocks=[[8,4,8],[8,4,8]]", "baseline.astgcn.chev_filter=8",
                                            "baseline.astgcn.time_filter=8"])
            m = build_baseline(cfg, P, W.n_cov_fut, np.linspace(0.02, 0.5, 7))
            out = m(tb)
            y = out["y"] if name in ("pag", "piast") else out["y_q"][..., 3]
            self.assertEqual(tuple(y.shape), (2, 24, 20), name)
            self.assertTrue(bool(torch.isfinite(y).all()), name)
        self.assertEqual(set(NAMES), set(ENCODERS) | {"pag", "piast"})


if __name__ == "__main__":
    unittest.main()
