"""Tests of equal predictive accuracy and resampling uncertainty for forecast errors."""
from __future__ import annotations

import numpy as np
from scipy import stats


def long_run_variance(d: np.ndarray, h: int) -> tuple[float, str]:
    """HAC variance of the mean-zero loss differential.

    Diebold & Mariano (1995): rectangular kernel truncated at h-1 lags, which is
    the correct autocovariance structure for optimal h-step errors (MA(h-1)).
    If that estimate is not positive, fall back to Bartlett (Newey-West)
    weights with the same truncation, which is guaranteed non-negative.
    """
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    gam = [np.dot(d[k:], d[: n - k]) / n for k in range(h)]
    lrv = gam[0] + 2.0 * sum(gam[1:])
    if lrv > 0:
        return float(lrv), "rectangular"
    lrv = gam[0] + 2.0 * sum((1 - k / h) * gam[k] for k in range(1, h))
    return float(lrv), "bartlett"


def diebold_mariano(e_model, e_bench, h: int, loss: str = "se") -> dict:
    """DM test with Harvey-Leybourne-Newbold (1997) small-sample correction.

    d_t = L(e_model) - L(e_bench); negative mean => model has lower loss.
    Two-sided p-value from Student t with n-1 degrees of freedom.
    """
    e1, e2 = np.asarray(e_model, float), np.asarray(e_bench, float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[ok], e2[ok]
    L = (lambda e: e**2) if loss == "se" else np.abs
    d = L(e1) - L(e2)
    n = len(d)
    out = {"n": n, "mean_loss_diff": float(np.mean(d)) if n else np.nan}
    if n < 2 or np.allclose(d, d[0]):
        return {**out, "dm_stat": np.nan, "hln_stat": np.nan, "p_value": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "kernel": "n/a"}
    lrv, kernel = long_run_variance(d, h)
    se = np.sqrt(lrv / n)
    dm = out["mean_loss_diff"] / se
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    hln = dm * adj
    p = 2 * stats.t.sf(abs(hln), df=n - 1)
    tcrit = stats.t.ppf(0.975, df=n - 1)
    return {
        **out,
        "dm_stat": float(dm),
        "hln_stat": float(hln),
        "p_value": float(p),
        "ci95_low": float(out["mean_loss_diff"] - tcrit * se / adj),
        "ci95_high": float(out["mean_loss_diff"] + tcrit * se / adj),
        "kernel": kernel,
    }


def holm(pvals) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values; NaNs are ignored and returned as NaN."""
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, j in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[j]))
        out[j] = running
    return out


def block_length(n: int, h: int) -> int:
    """At least h (MA(h-1) overlap) and at least n^(1/3) (rule of thumb)."""
    return int(max(h, np.ceil(n ** (1 / 3))))


def moving_block_indices(n: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """Circular moving-block bootstrap index vector of length n."""
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=k)
    idx = (starts[:, None] + np.arange(b)[None, :]) % n
    return idx.ravel()[:n]


def bootstrap_skill(e_model, e_bench, h: int, reps: int, rng: np.random.Generator) -> dict:
    """Block-bootstrap distribution of RMSE skill = 1 - RMSE_model / RMSE_bench and of the mean SE difference."""
    e1, e2 = np.asarray(e_model, float), np.asarray(e_bench, float)
    ok = np.isfinite(e1) & np.isfinite(e2)
    e1, e2 = e1[ok], e2[ok]
    n = len(e1)
    b = block_length(n, h)
    skill = 1 - np.sqrt(np.mean(e1**2)) / np.sqrt(np.mean(e2**2))
    i = block_index_matrix(n, b, reps, rng)
    s1, s2 = (e1**2)[i], (e2**2)[i]
    sk = 1 - np.sqrt(s1.mean(1)) / np.sqrt(s2.mean(1))
    md = (s1 - s2).mean(1)
    return {
        "n": n,
        "block_length": b,
        "skill_rmse": float(skill),
        "skill_ci95_low": float(np.quantile(sk, 0.025)),
        "skill_ci95_high": float(np.quantile(sk, 0.975)),
        "prob_skill_positive_boot": float(np.mean(sk > 0)),
        "mse_diff_boot_ci95_low": float(np.quantile(md, 0.025)),
        "mse_diff_boot_ci95_high": float(np.quantile(md, 0.975)),
    }


def bootstrap_rmse_ci(e, h: int, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    e = np.asarray(e, float)
    e = e[np.isfinite(e)]
    n = len(e)
    vals = np.sqrt((e**2)[block_index_matrix(n, block_length(n, h), reps, rng)].mean(1))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def block_index_matrix(n: int, b: int, reps: int, rng: np.random.Generator) -> np.ndarray:
    """reps x n matrix of circular moving-block bootstrap indices."""
    k = int(np.ceil(n / b))
    starts = rng.integers(0, n, size=(reps, k))
    idx = (starts[:, :, None] + np.arange(b)[None, None, :]) % n
    return idx.reshape(reps, -1)[:, :n]
