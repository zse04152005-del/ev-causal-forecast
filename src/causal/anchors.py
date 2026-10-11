"""锚定文件（阶段 6 的输出 = 模型的输入，设计文档 6.1、6.4、6.5、6.7）。

JSON 格式：
{
  "version": "...", "source": "...", "target": "utilization", "price": "total",
  "estimand": "window_3h_log_mean", "design": "C",
  "own":   [{"group": 1, "context": "weekday_day", "beta": -0.5, "se": 0.2, "n_zones": 37, "n_events": 2329,
             "level": "cell"}, ...],             # group / context 可写 "all"
  "cross": [{"ring": "0-2km", "delta": 0.0, "se": 0.05}, ...],
  "shift": {"gamma": 0.0, "se": 1.0},
  "lag_weights": null,
  "window_factor": null            # 可选：有滞后核时"锚定值 ÷ 长期弹性"的经验比例（例如由 5 分钟半合成验证给出）；
                                   # 省略则按 lag_weights 与孤立切换的理想路径计算
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..data.calendar import CONTEXT_NAMES

PLACEHOLDER_SOURCES = {"placeholder"}


@dataclass
class AnchorSet:
    beta: np.ndarray                 # [G, C]
    se: np.ndarray                   # [G, C]
    delta: np.ndarray                # [K]
    delta_se: np.ndarray             # [K]
    gamma: float = 0.0
    gamma_se: float = 1.0
    lag_weights: np.ndarray | None = None
    source: str = ""
    provenance: np.ndarray | None = None   # [G, C] 字符串：该格子取自哪个层级
    meta: dict = field(default_factory=dict)

    @property
    def is_placeholder(self) -> bool:
        return self.source in PLACEHOLDER_SOURCES

    def window_factor(self, window: int = 3) -> float:
        return window_factor(self.lag_weights, window)


def window_factor(lag_weights, window: int = 3) -> float:
    """口径对齐（设计文档 6.5）：切换后 window 小时内的平均累计调整比例。静态模型为 1。"""
    if lag_weights is None:
        return 1.0
    w = np.asarray(lag_weights, dtype=np.float64)
    w = w / w.sum()
    cum = np.cumsum(w)
    return float(np.mean([cum[min(k, len(cum) - 1)] for k in range(window)]))


def project_monotone_nonneg(delta: np.ndarray) -> np.ndarray:
    """投影到 δ1 ≥ δ2 ≥ … ≥ 0（相邻违例合并，PAV）。"""
    v = [[float(x), 1] for x in delta]
    out = []
    for val, w in v:
        out.append([val, w])
        while len(out) > 1 and out[-2][0] < out[-1][0]:      # 需要非增
            a, b = out.pop(), out.pop()
            out.append([(a[0] * a[1] + b[0] * b[1]) / (a[1] + b[1]), a[1] + b[1]])
    res = []
    for val, w in out:
        res += [max(val, 0.0)] * w
    return np.array(res)


def _ctx_index(c) -> int | None:
    if c == "all":
        return None
    return CONTEXT_NAMES.index(c) if isinstance(c, str) else int(c)


def load_anchors(path: str, G: int, C: int = 4, K: int = 3) -> AnchorSet:
    with open(path, "r", encoding="utf-8") as f:
        js = json.load(f)
    beta = np.full((G, C), np.nan)
    se = np.full((G, C), np.nan)
    prov = np.full((G, C), "", dtype=object)
    # 先填宽泛层级，再用更细的层级覆盖：全市 → 情境 → 功能区 → 格子
    def rank(e):
        return (e.get("group", "all") != "all") * 2 + (e.get("context", "all") != "all")
    for e in sorted(js.get("own", []), key=rank):
        gs = range(G) if e.get("group", "all") == "all" else [int(e["group"])]
        ci = _ctx_index(e.get("context", "all"))
        cs = range(C) if ci is None else [ci]
        for g in gs:
            for c in cs:
                if g < G:
                    beta[g, c], se[g, c] = e["beta"], e["se"]
                    prov[g, c] = e.get("level", f"{e.get('group', 'all')}|{e.get('context', 'all')}")
    if np.isnan(beta).any():
        raise ValueError(f"锚定文件 {path} 没有覆盖所有 功能区×情境 格子")
    if (beta > 0).any():
        import warnings
        warnings.warn(f"锚定文件 {path} 中有 {int((beta > 0).sum())} 个格子弹性为正（与需求定律相反）；截断反馈会原样使用，"
                      "正式锚定前应按合并规则并入上一层级", stacklevel=2)
    cross = js.get("cross", [])
    delta = np.array([e["delta"] for e in cross][:K] + [0.0] * max(0, K - len(cross)))
    dse = np.array([e["se"] for e in cross][:K] + [1.0] * max(0, K - len(cross)))
    sh = js.get("shift") or {}
    lw = js.get("lag_weights")
    return AnchorSet(beta=beta, se=se, delta=delta, delta_se=dse, gamma=float(sh.get("gamma", 0.0)),
                     gamma_se=float(sh.get("se", 1.0)), lag_weights=None if lw is None else np.asarray(lw, float),
                     source=js.get("source", ""), provenance=prov,
                     meta={k: js.get(k) for k in ("version", "target", "price", "estimand", "design", "window_factor")})


def placeholder_json(beta: float = -0.45, se: float = 0.5, K: int = 3) -> dict:
    return {
        "version": "placeholder", "source": "placeholder", "target": "any", "price": "total",
        "estimand": "window_3h_log_mean", "design": "none",
        "note": "阶段 6 之前的占位锚定：β̂ = PIAST 参考值 −0.45，se 很宽；评价脚本遇到它会拒绝输出论文结果",
        "own": [{"group": "all", "context": "all", "beta": beta, "se": se, "level": "placeholder"}],
        "cross": [{"ring": f"ring{k + 1}", "delta": 0.0, "se": 0.2} for k in range(K)],
        "shift": {"gamma": 0.0, "se": 1.0},
        "lag_weights": None,
    }


def merge_levels(est, G: int, C: int = 4, min_zones: int = 10, max_se: float = 0.5, require_negative: bool = False) -> list:
    """把 estimate_cells 的三层结果合并成最终的格子表（设计文档 6.4 合并规则）。返回 own 条目列表。"""
    def ok(r):
        good = r is not None and np.isfinite(r["beta"]) and np.isfinite(r["se"]) and \
            r["n_zones"] >= min_zones and r["se"] <= max_se
        return good and (not require_negative or r["beta"] <= 0)      # require_negative：正号弹性与需求定律相反，并入上一层级
    cell = {tuple(r["key"]): r for _, r in est[est.level == "cell"].iterrows()}
    ctx = {int(r["key"]): r for _, r in est[est.level == "context"].iterrows()}
    city = est[est.level == "city"].iloc[0]
    own = []
    for g in range(G):
        for c in range(C):
            r, lvl = cell.get((g, c)), "cell"
            if not ok(r):
                r, lvl = ctx.get(c), "context"
                if not ok(r):
                    r, lvl = city, "city"
            own.append({"group": g, "context": CONTEXT_NAMES[c], "beta": float(r["beta"]), "se": float(r["se"]),
                        "n_zones": int(r["n_zones"]), "n_events": int(r["n_events"]), "level": lvl})
    return own


def save_json(obj: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
