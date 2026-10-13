"""Gradient-boosted trees with lagged demand: the standard strong baseline of energy forecasting (second review round).

One HistGradientBoostingRegressor with the quantile loss per quantile level (scikit-learn; no PyTorch, no LightGBM
needed), shared by all zones and horizons. Each training row is (forecast origin t0, zone i, horizon h) with features
    the last L values of the utilisation of zone i and the "unchanged" flag at t0,
    the horizon h, the calendar (6) and weather (6) of the target hour t0 + h,
    the static attributes of zone i and its training-period mean utilisation per hour of week (seasonal level);
    zones are identified through these attributes (scikit-learn limits categorical features to 255 levels).
The model is price-blind, like the reference forecasters of Table 2. Training rows are subsampled over origins
(gbdt.origin_stride) to bound memory; rows with an invalid (imputed) target are dropped, as in every other model.
Interface as NaiveForecaster: fit(P, W_train, W_val) and predict(P, W) -> [n, H, N, Q].
"""
from __future__ import annotations

import numpy as np

from ..data.dataset import Prepared
from ..data.windows import WindowDataset


class GBDTForecaster:
    def __init__(self, quantiles, clip_max: float | None = 1.0, origin_stride: int = 6, max_iter: int = 400,
                 learning_rate: float = 0.05, max_leaf_nodes: int = 63, min_samples_leaf: int = 200, seed: int = 0):
        self.quantiles = np.asarray(quantiles, dtype=float)
        self.clip_max = clip_max
        self.stride, self.seed = int(origin_stride), int(seed)
        self.params = dict(max_iter=int(max_iter), learning_rate=float(learning_rate),
                           max_leaf_nodes=int(max_leaf_nodes), min_samples_leaf=int(min_samples_leaf))
        self.models = []
        self.prof = None

    # ------------------------------------------------------------ features
    def _features(self, P: Prepared, t0: np.ndarray, L: int, H: int) -> np.ndarray:
        """[len(t0) * H * N, F] in the order (origin, horizon, zone)."""
        n, N = len(t0), P.N
        hist = t0[:, None] + np.arange(-L + 1, 1)[None, :]                     # [n, L]
        fut = t0[:, None] + np.arange(1, H + 1)[None, :]                       # [n, H]
        lags = P.y[hist].transpose(0, 2, 1)                                    # [n, N, L]
        stale = P.stale[t0]                                                    # [n, N]
        how = (P.dow * 24 + P.tod)[fut]                                        # [n, H]
        cols = []
        rep = lambda a: np.broadcast_to(a, (n, H, N) + a.shape[3:])            # noqa: E731
        cols.append(rep(lags[:, None, :, :]).reshape(n * H * N, L))
        cols.append(rep(stale[:, None, :, None]).reshape(-1, 1))
        cols.append(rep(np.arange(1, H + 1)[None, :, None, None].astype(float)).reshape(-1, 1))
        cal = np.concatenate([P.cal[fut], P.weather[fut]], axis=-1)            # [n, H, 12]
        cols.append(rep(cal[:, :, None, :]).reshape(-1, cal.shape[-1]))
        cols.append(rep(P.static[None, None, :, :]).reshape(-1, P.static.shape[1]))
        cols.append(rep(self.prof[how][..., None]).reshape(-1, 1))             # seasonal level of the target hour
        return np.concatenate(cols, axis=1).astype(np.float32)

    def _targets(self, P: Prepared, t0: np.ndarray, H: int):
        fut = t0[:, None] + np.arange(1, H + 1)[None, :]
        return P.y[fut].reshape(-1), P.valid[fut].reshape(-1)

    # ------------------------------------------------------------ fit / predict
    def fit(self, P: Prepared, W_train: WindowDataset, W_val: WindowDataset | None = None) -> "GBDTForecaster":
        from sklearn.ensemble import HistGradientBoostingRegressor
        tr = P.split.train_end
        how = (P.dow * 24 + P.tod)[: tr + 1]
        y, v = P.y[: tr + 1], P.valid[: tr + 1]
        prof = np.zeros((168, P.N))
        for k in range(168):
            m = how == k
            den = v[m].sum(0)
            prof[k] = np.divide((y[m] * v[m]).sum(0), den, out=np.zeros(P.N), where=den > 0)
        self.prof = prof
        t0 = W_train.origins[:: self.stride]
        X = self._features(P, t0, W_train.L, W_train.H)
        yy, vv = self._targets(P, t0, W_train.H)
        X, yy = X[vv], yy[vv]
        self.models = []
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=float(q), early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                              random_state=self.seed, **self.params)
            self.models.append(m.fit(X, yy))
        return self

    def predict(self, P: Prepared, W: WindowDataset, chunk: int = 64) -> np.ndarray:
        n, H, N = len(W.origins), W.H, P.N
        out = np.empty((n, H, N, len(self.quantiles)), dtype=np.float32)
        for s in range(0, n, chunk):
            t0 = W.origins[s: s + chunk]
            X = self._features(P, t0, W.L, H)
            pq = np.stack([m.predict(X) for m in self.models], axis=-1).reshape(len(t0), H, N, -1)
            out[s: s + chunk] = pq
        out = np.maximum(out, 0.0)
        if self.clip_max is not None:
            out = np.minimum(out, self.clip_max)
        return np.sort(out, axis=-1)
