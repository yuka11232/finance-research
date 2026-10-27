"""Small-sample comparisons of household groups (independent observations).

Survey respondents are treated as independent draws from an unknown
convenience-sample population; no design weights exist, so all statistics
describe the surveyed households only.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def describe(x, rng: np.random.Generator, reps: int) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0}
    out = {"n": n, "mean": float(x.mean()), "median": float(np.median(x)), "sd": float(x.std(ddof=1)) if n > 1 else np.nan}
    if n >= 3:
        bs = rng.choice(x, size=(reps, n), replace=True)
        out.update(
            mean_ci95_low=float(np.quantile(bs.mean(1), 0.025)),
            mean_ci95_high=float(np.quantile(bs.mean(1), 0.975)),
            median_ci95_low=float(np.quantile(np.median(bs, 1), 0.025)),
            median_ci95_high=float(np.quantile(np.median(bs, 1), 0.975)),
        )
    return out


def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(j * (a.mean() - b.mean()) / sp)


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def permutation_test(a, b, rng: np.random.Generator, reps: int = 20000) -> float:
    """Two-sided permutation p-value for the difference in means (with +1 correction)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    na = len(a)
    obs = abs(a.mean() - b.mean())
    perm = np.argsort(rng.random((reps, len(pooled))), axis=1)  # each row is a random permutation
    x = pooled[perm]
    diffs = np.abs(x[:, :na].mean(1) - x[:, na:].mean(1))
    cnt = int(np.sum(diffs >= obs - 1e-12))
    return float((cnt + 1) / (reps + 1))


def bootstrap_diff_ci(a, b, rng: np.random.Generator, reps: int) -> tuple[float, float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = rng.choice(a, (reps, len(a))).mean(1) - rng.choice(b, (reps, len(b))).mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def mde_two_sample(sd_a: float, sd_b: float, n_a: int, n_b: int, alpha=0.05, power=0.80) -> float:
    """Minimum detectable difference in means (normal approximation, unequal variances)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(sd_a**2 / n_a + sd_b**2 / n_b))


def compare(a, b, rng: np.random.Generator, reps: int, min_n: int = 5) -> dict:
    """Welch t, Mann-Whitney U, permutation test, effect sizes, bootstrap CI and MDE."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    out = {"n_a": len(a), "n_b": len(b)}
    if len(a) < min_n or len(b) < min_n:
        return {**out, "tested": False, "reason": f"group size < {min_n}"}
    lo, hi = bootstrap_diff_ci(a, b, rng, reps)
    return {
        **out,
        "tested": True,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff_mean": float(a.mean() - b.mean()),
        "diff_median": float(np.median(a) - np.median(b)),
        "diff_ci95_low_boot": lo,
        "diff_ci95_high_boot": hi,
        "welch_p": float(stats.ttest_ind(a, b, equal_var=False).pvalue),
        "mannwhitney_p": float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue),
        "permutation_p": permutation_test(a, b, rng),
        "hedges_g": hedges_g(a, b),
        "cliffs_delta": cliffs_delta(a, b),
        "mde_80pct_power": mde_two_sample(a.std(ddof=1), b.std(ddof=1), len(a), len(b)),
    }
