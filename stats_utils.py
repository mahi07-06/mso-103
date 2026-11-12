"""
stats_utils.py - the handful of statistical functions the model needs,
written with numpy only (so the project has no scipy dependency).
"""
import math
import numpy as np


def gamma_cdf(x, shape):
    """Regularised lower incomplete gamma P(shape, x) = CDF of Gamma(shape, scale=1)."""
    if x <= 0:
        return 0.0
    lg = math.lgamma(shape)
    if x < shape + 1.0:                       # series expansion
        term = total = 1.0 / shape
        a = shape
        for _ in range(1000):
            a += 1.0
            term *= x / a
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return total * math.exp(-x + shape * math.log(x) - lg)
    tiny = 1e-300                             # continued fraction (Lentz)
    b = x + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return 1.0 - math.exp(-x + shape * math.log(x) - lg) * h


def gamma_ppf(p, shape, scale=1.0):
    """Inverse CDF G^-1(p) of Gamma(shape, scale) by bisection (Excel: GAMMA.INV)."""
    lo, hi = 0.0, max(1.0, 2.0 * shape)
    while gamma_cdf(hi, shape) < p:
        hi *= 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gamma_cdf(mid, shape) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) * scale


def gamma_pdf(x, shape):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = np.exp((shape - 1) * np.log(x[pos]) - x[pos] - math.lgamma(shape))
    return out


def ks_test(sample, cdf):
    """One-sample Kolmogorov-Smirnov test. Returns (D statistic, asymptotic p-value)."""
    x = np.sort(np.asarray(sample, dtype=float))
    n = len(x)
    f = np.array([cdf(v) for v in x])
    d = max(np.max(np.arange(1, n + 1) / n - f), np.max(f - np.arange(0, n) / n))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if lam < 0.2:
        return d, 1.0
    p = 2.0 * sum((-1) ** (j - 1) * math.exp(-2.0 * j * j * lam * lam) for j in range(1, 101))
    return d, min(1.0, max(0.0, p))
