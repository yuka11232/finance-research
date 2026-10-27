"""First-order fuzzy time-series models.

Implements, from the primary sources (full references in the repository README):

* Huarng (2001) average- and distribution-based interval lengths, exactly as
  summarised in Yu (2005, Sect. 2.2), with the "Map Table" generalised to
  powers of ten (base = 10^k for lengths in (10^k, 10^(k+1)]), which
  reproduces every row of Huarng's table (0.1-1 -> 0.1, 1.1-10 -> 1,
  11-100 -> 10, 101-1000 -> 100).
* Chen (1996) conventional model (Yu 2005, Sect. 2.3): FLR groups with
  repeated relationships counted once, arithmetic mean of midpoints.
* Yu (2005) weighted model (Sect. 4, Eqs. 6-13): FLR groups keep recurrent
  relationships in chronological order; weights w_i = i (Eq. 10) normalised
  to sum to one (Eq. 8); final(t) = M(t) x W(t)^T (Eq. 13).
* An equal-weight recurrence model (recurrence kept, every FLR weighted
  equally). Yu notes (Sect. 3.2) that unweighted models are the equal-weight
  special case; comparing it with the weighted model isolates the effect of
  the chronological weights, and comparing it with Chen isolates recurrence.

Fuzzy sets follow Chen/Yu: A_i = ... 0.5/u_{i-1} + 1/u_i + 0.5/u_{i+1} ...,
so a datum's maximal membership is in the set whose core interval contains
it, and fuzzification reduces to locating the interval.

Documented extensions (not specified in Yu 2005):
  * Multi-step forecasts are recursive: the defuzzified h-step forecast is
    fuzzified and used as the state for step h+1; FLR groups are not updated
    with forecasts. Yu evaluates one-step forecasts only.
  * Observations outside the universe of discourse are assigned to the
    nearest end interval.
  * An LHS with no FLR group forecasts its own midpoint (Chen's Rule 1;
    Yu does not restate this rule).
  * Universe start: Yu rounds the sample minimum down to the nearest hundred
    for TAIEX (length base 10). We round down to a multiple of 10 x base,
    which reproduces that rule, and extend the end so the maximum is covered.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

METHODS = ("chen", "yu_equal", "yu_weighted")


def map_table_base(length: float) -> float:
    """Huarng (2001) base for a given length, generalised to powers of ten."""
    if length <= 0 or not np.isfinite(length):
        raise ValueError(f"length must be positive, got {length}")
    return 10.0 ** (math.ceil(math.log10(length) - 1e-12) - 1)


def average_based_length(x: np.ndarray) -> float:
    d = np.abs(np.diff(np.asarray(x, dtype=float)))
    half = d.mean() / 2.0
    base = map_table_base(half)
    length = math.floor(half / base + 1e-9) * base
    return max(length, base)


def distribution_based_length(x: np.ndarray) -> float:
    d = np.abs(np.diff(np.asarray(x, dtype=float)))
    base = map_table_base(d.mean())
    best = base
    k = 1
    while True:
        cand = k * base
        if np.sum(d > cand + 1e-12) >= len(d) / 2.0:
            best = cand
            k += 1
        else:
            break
    return best


@dataclass
class Universe:
    start: float
    length: float
    n_intervals: int

    @classmethod
    def from_data(cls, x: np.ndarray, length: float) -> "Universe":
        lo, hi = float(np.min(x)), float(np.max(x))
        unit = 10.0 * map_table_base(length)
        start = math.floor(lo / unit + 1e-12) * unit
        n = max(1, math.floor((hi - start) / length + 1e-12) + 1)
        return cls(start=start, length=length, n_intervals=n)

    @property
    def midpoints(self) -> np.ndarray:
        return self.start + (np.arange(self.n_intervals) + 0.5) * self.length

    def fuzzify(self, x) -> np.ndarray:
        idx = np.floor((np.asarray(x, dtype=float) - self.start) / self.length + 1e-12).astype(int)
        return np.clip(idx, 0, self.n_intervals - 1)


@dataclass
class FuzzyTimeSeries:
    """Fit on a 1-D array, then forecast one or several steps ahead."""

    method: str = "yu_weighted"
    interval: str = "average"  # "average" | "distribution" | float length
    universe: Universe | None = field(default=None, init=False)
    groups: dict = field(default_factory=dict, init=False)
    last_state: int | None = field(default=None, init=False)

    def fit(self, x) -> "FuzzyTimeSeries":
        if self.method not in METHODS:
            raise ValueError(self.method)
        x = np.asarray(x, dtype=float)
        if isinstance(self.interval, (int, float)):
            length = float(self.interval)
        elif self.interval == "average":
            length = average_based_length(x)
        elif self.interval == "distribution":
            length = distribution_based_length(x)
        else:
            raise ValueError(self.interval)
        self.universe = Universe.from_data(x, length)
        states = self.universe.fuzzify(x)
        groups: dict[int, list[int]] = {}
        for lhs, rhs in zip(states[:-1], states[1:]):  # chronological FLRs
            groups.setdefault(int(lhs), []).append(int(rhs))
        self.groups = groups
        self.last_state = int(states[-1])
        return self

    def _defuzzify(self, state: int) -> float:
        m = self.universe.midpoints
        rhs = self.groups.get(state)
        if not rhs:
            return float(m[state])  # Chen Rule 1
        if self.method == "chen":
            uniq = sorted(set(rhs))  # repeated FLRs counted once
            return float(np.mean(m[uniq]))
        mids = m[np.asarray(rhs)]
        if self.method == "yu_equal":
            return float(np.mean(mids))
        w = np.arange(1, len(rhs) + 1, dtype=float)  # Eq. (10): w_i = i
        w /= w.sum()  # Eq. (8)
        return float(mids @ w)  # Eq. (13)

    def predict_next(self, state: int) -> float:
        return self._defuzzify(state)

    def forecast(self, steps: int) -> np.ndarray:
        out = np.empty(steps)
        state = self.last_state
        for h in range(steps):
            out[h] = self._defuzzify(state)
            state = int(self.universe.fuzzify(out[h]))
        return out

    def one_step_in_sample(self, x) -> np.ndarray:
        """One-step forecasts for x[1:], using actual previous states (Yu's evaluation)."""
        states = self.universe.fuzzify(np.asarray(x, dtype=float))
        return np.array([self._defuzzify(int(s)) for s in states[:-1]])
