"""Exact baseline8 Jeffreys anchors and frozen training quantization."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import betainc, betaincinv
from scipy.stats import beta

from ._grades import GRADES

FEATURES: list[str] = [
    'depth',
    'opposite_depth',
    'p_raw',
    'width',
    'posterior_sd',
    'D',
    'standardized_difference',
    'opposite_missing',
]
DELTA: float = 0.10
PROTOCOL: str = 'baseline8_jeffreys_delta010_ridge_v1'


@lru_cache(maxsize=200000)
def strand_risk(m: int, u: int, om: int, ou: int, delta: float) -> tuple[float, float]:
    # Integrate in posterior probability coordinates: no endpoint beta-density singularities.
    """Integrate the Jeffreys strand-disagreement probability with adaptive quadrature."""
    previous = None
    for points in (32, 64, 128, 256, 512):
        nodes, weights = quadrature(points)
        theta = betaincinv(m + 0.5, u + 0.5, nodes)
        value = float(
            np.dot(
                weights,
                betainc(om + 0.5, ou + 0.5, np.clip(theta - delta, 0, 1))
                + 1
                - betainc(om + 0.5, ou + 0.5, np.clip(theta + delta, 0, 1)),
            )
        )
        error = abs(value - previous) if previous is not None else 1.0
        if error < 2e-5:
            return min(1.0, max(0.0, value)), error
        previous = value
    return min(1.0, max(0.0, value)), error


@lru_cache(maxsize=5)
def quadrature(points: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return cached Gauss-Legendre nodes and weights on the probability interval."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    return (nodes + 1) / 2, weights / 2


def features(
    data: dict[str, npt.NDArray[Any]], delta: float, anchors: bool = True
) -> dict[str, npt.NDArray[Any]]:
    """Compute baseline8 count features and optionally exact posterior risk anchors."""
    n, m, on, om = (data[k].astype(float) for k in ('depth', 'm', 'opposite_depth', 'om'))
    u = data.get('u', n - m).astype(float)
    valid = (
        (data['input_legal'] == 1)
        & (n > 0)
        & (data['strand'] != '?')
        & np.isfinite(m)
        & np.isfinite(u)
        & np.isfinite(n)
        & (m >= 0)
        & (u >= 0)
        & (m + u == n)
        & (m == np.floor(m))
        & (u == np.floor(u))
    )
    p = np.divide(m, n, out=np.full(len(n), np.nan), where=valid)
    opposite = (on > 0) & np.isfinite(on)
    op = np.divide(om, on, out=np.full(len(n), np.nan), where=opposite)
    sd = np.sqrt((m + 0.5) * (n - m + 0.5) / ((n + 1) ** 2 * (n + 2)))
    osd = np.sqrt((om + 0.5) * (on - om + 0.5) / ((on + 1) ** 2 * (on + 2)))
    width = np.full(len(n), np.nan)
    width[valid] = beta.ppf(0.975, m[valid] + 0.5, n[valid] - m[valid] + 0.5) - beta.ppf(
        0.025, m[valid] + 0.5, n[valid] - m[valid] + 0.5
    )
    result = {
        **data,
        'valid': valid,
        'p_raw': p,
        'width': width,
        'posterior_sd': sd,
        'D': np.abs(p - op),
        'standardized_difference': np.abs(p - op) / np.sqrt(sd**2 + osd**2),
        'opposite_missing': (~opposite).astype(float),
    }
    if anchors:
        rc = np.full(len(n), np.nan)
        rc[valid] = (
            betainc(m[valid] + 0.5, n[valid] - m[valid] + 0.5, np.clip(p[valid] - delta, 0, 1))
            + 1
            - betainc(m[valid] + 0.5, n[valid] - m[valid] + 0.5, np.clip(p[valid] + delta, 0, 1))
        )
        rs = np.full(len(n), np.nan)
        error = np.full(len(n), np.nan)
        for i in np.flatnonzero(valid & opposite):
            left, right = sorted([(int(m[i]), int(n[i] - m[i])), (int(om[i]), int(on[i] - om[i]))])
            rs[i], error[i] = strand_risk(*left, *right, delta)
        result.update(
            r_count=rc,
            r_strand=rs,
            integration_error=error,
            r_anchor=np.where(opposite, np.maximum(rc, rs), rc),
        )
    return result


def fit_quantizer(
    risk: npt.NDArray[np.float64], blocks: npt.NDArray[np.str_], bins: int = 128
) -> dict[str, Any]:
    """Freeze ordered risk representatives and four spatially supported grades."""
    risk = np.asarray(risk, dtype=float)
    blocks = np.asarray(blocks)
    if risk.ndim != 1 or blocks.shape != risk.shape or not len(risk):
        raise ValueError('Risk and blocks must be nonempty matching vectors')
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError('Risk must be finite and within [0, 1]')
    if not 4 <= bins <= 128:
        raise ValueError('Use 4 to 128 finite risk bins')
    unique = np.unique(risk)
    if len(unique) <= bins:
        edges = unique[:-1] + np.diff(unique) / 2
        edges = np.where(edges <= unique[:-1], unique[1:], edges)
    else:
        edges = np.unique(np.quantile(risk, np.arange(1, bins) / bins))
        edges = edges[(edges > risk.min()) & (edges < risk.max())]
    index = np.searchsorted(edges, risk, side='right')
    occupied = np.unique(index)
    # Remove boundaries around empty bins before freezing the representation.
    edges = edges[occupied[1:] - 1]
    index = np.searchsorted(edges, risk, side='right')
    size = len(edges) + 1
    counts = np.bincount(index, minlength=size)
    values = np.empty(size, dtype=float)
    for i in range(size):
        selected = risk[index == i]
        origin = selected.min()
        values[i] = origin + np.mean(selected - origin)
    if np.any(np.diff(values) <= 0):
        raise ValueError('Finite risk representatives must increase strictly')
    quantized = values[index]
    result: dict[str, Any] = {
        'protocol': 'training_quantile_128_max_ordered_risk_v1',
        'edges': edges.tolist(),
        'values': values.tolist(),
        'training_rows': len(risk),
        'training_quantization_mse': float(np.mean((quantized - risk) ** 2)),
        'training_max_absolute_quantization_error': float(np.max(np.abs(quantized - risk))),
        'cuts': [],
        'supported': False,
    }
    if size < 4:
        result['reason'] = 'fewer_than_four_risks'
        return result
    weights = np.r_[0, np.cumsum(counts)]
    sums = np.r_[0.0, np.cumsum(counts * values)]
    squares = np.r_[0.0, np.cumsum(counts * values**2)]
    # A segment has at least two blocks unless every constituent bin contains
    # the same single block. Precompute this condition without scanning rows.
    single_block: list[Any] = []
    for i in range(size):
        present = np.unique(blocks[index == i])
        single_block.append(present[0] if len(present) == 1 else None)
    allowed = np.zeros((size, size + 1), dtype=bool)
    for start in range(size):
        first = single_block[start]
        multiple = first is None
        for end in range(start + 1, size + 1):
            multiple = multiple or single_block[end - 1] is None or single_block[end - 1] != first
            allowed[start, end] = multiple and weights[end] - weights[start] >= 0.01 * len(risk)
    dp = np.full((5, size + 1), np.inf)
    previous = np.full((5, size + 1), -1, dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, 5):
        for end in range(k, size + 1):
            for start in range(k - 1, end):
                if not allowed[start, end]:
                    continue
                count = weights[end] - weights[start]
                loss = max(
                    0.0, squares[end] - squares[start] - (sums[end] - sums[start]) ** 2 / count
                )
                objective = dp[k - 1, start] + loss
                if objective < dp[k, end]:
                    dp[k, end], previous[k, end] = objective, start
    if not np.isfinite(dp[4, size]):
        result['reason'] = 'insufficient_grade_count_or_blocks'
        return result
    cuts: list[float] = []
    end = size
    for k in range(4, 0, -1):
        start = previous[k, end]
        if start:
            boundary = values[start - 1] + (values[start] - values[start - 1]) / 2
            cuts.append(float(values[start] if boundary <= values[start - 1] else boundary))
        end = start
    result.update(cuts=sorted(cuts), supported=True, grade_loss=float(dp[4, size]))
    return result


def apply_quantizer(
    risk: npt.NDArray[np.float64], frozen: dict[str, Any]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.str_]]:
    """Map risk values through frozen representatives and H/MH/ML/L boundaries."""
    risk = np.asarray(risk, dtype=float)
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError('Risk must be finite and within [0, 1]')
    values = np.asarray(frozen['values'], dtype=float)
    quantized = values[np.searchsorted(frozen['edges'], risk, side='right')]
    if frozen['supported']:
        grades = np.asarray(list(GRADES))[np.searchsorted(frozen['cuts'], quantized, side='right')]
    else:
        grades = np.full(risk.shape, 'U')
    return quantized, grades
