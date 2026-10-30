"""Fold-isolated Ridge, deterministic capped sampling and portable parameters."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ._science import DELTA, FEATURES, PROTOCOL, fit_quantizer

LOGGER: logging.Logger = logging.getLogger(__name__)


def gpu_modules() -> tuple[Any, Any]:
    """Load usable CuPy/cuML lazily; reject unavailable CUDA without CPU fallback."""
    try:
        import cuml
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device')
        return cp, cuml
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            'GPU explicitly requested: usable cuML/CuPy/CUDA required; no CPU fallback'
        ) from exc


def stable_keys(
    chrom: str, starts: npt.NDArray[np.int64], seed: int = 1729
) -> npt.NDArray[np.uint64]:
    """Assign reproducible NumPy random keys in actual-coordinate order."""
    # Encode signed seeds without process-dependent Python hashes.
    normalized_seed = 2 * seed if seed >= 0 else -2 * seed - 1
    rng = np.random.default_rng(
        np.random.SeedSequence([normalized_seed, *chrom.encode('utf-8')])
    )
    order = np.argsort(starts, kind='stable')
    keys = np.empty(len(starts), dtype=np.uint64)
    keys[order] = rng.bit_generator.random_raw(len(starts))
    return keys


def fit_ridge(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], alpha: float, gpu: bool
) -> dict[str, Any]:
    """Fit fold-local median imputation, scaling and CPU/GPU SVD Ridge parameters."""
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    z = scaler.fit_transform(imputer.fit_transform(x)).astype(np.float64)
    if gpu:
        cp, cuml = gpu_modules()
        ridge = cuml.linear_model.Ridge(
            alpha=alpha, fit_intercept=True, solver='svd', output_type='numpy'
        )
        ridge.fit(z, y.astype(np.float64))
        coef = np.asarray(ridge.coef_, dtype=float)
        intercept = float(np.asarray(ridge.intercept_))
        backend = 'cuml-' + cuml.__version__
    else:
        import sklearn

        ridge = Ridge(alpha=alpha, fit_intercept=True, solver='svd')
        ridge.fit(z, y)
        coef, intercept = ridge.coef_, float(ridge.intercept_)
        backend = 'sklearn-' + sklearn.__version__
    return {
        'fill': imputer.statistics_.tolist(),
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist(),
        'coef': coef.tolist(),
        'intercept': intercept,
        'alpha': alpha,
        'backend': backend,
    }


def predict(
    x: npt.NDArray[np.float64], model: dict[str, Any], gpu: bool = False, *, clip: bool = True
) -> npt.NDArray[np.float64]:
    """Apply portable frozen preprocessing and coefficients to a complete matrix."""
    z = (
        np.where(np.isnan(x), np.asarray(model['fill']), x) - np.asarray(model['mean'])
    ) / np.asarray(model['scale'])
    if gpu:
        cp, _ = gpu_modules()
        # One device prediction for the entire chromosome; portable frozen coefficients.
        result = cp.asnumpy(
            cp.asarray(z, dtype=cp.float64) @ cp.asarray(model['coef'], dtype=cp.float64)
            + model['intercept']
        )
    else:
        result = z @ np.asarray(model['coef']) + model['intercept']
    return np.clip(result, 0.0, 1.0) if clip else result


def spatial_folds(blocks: npt.NDArray[np.str_]) -> npt.NDArray[np.int64]:
    """Assign real genomic blocks to deterministic folds, never splitting a block."""
    occupied = sorted(set(map(str, blocks)))
    # Research block hash on all-contig stable order; tuple keys avoid block collisions.
    contigs = sorted({block.split('\0')[0] for block in occupied})
    ranks = {chrom: i for i, chrom in enumerate(contigs)}
    lookup = {
        block: (
            (
                ranks[block.split('\0')[0]] * 1000
                + (int(block.split('\0')[1]) if '\0' in block else 0)
            )
            * 2654435761
            + 1729
        )
        % 4
        for block in occupied
    }
    if len(set(lookup.values())) < 4:
        # Small inputs: occupied real blocks, not rows, deterministically occupy <=4 folds.
        lookup = {block: i % min(4, len(occupied)) for i, block in enumerate(occupied)}
    return np.asarray([lookup[str(block)] for block in blocks], dtype=int)


def train(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    blocks: npt.NDArray[np.str_],
    gpu: bool,
    max_train: int,
    seed: int,
) -> dict[str, Any]:
    """Select Ridge alpha by spatial one-SE CV and freeze training-only quantization."""
    if not len(y):
        raise ValueError('No valid positive-depth CpG training records')
    folds = spatial_folds(blocks)
    LOGGER.info(
        'Fitting Ridge with %d training rows across %d spatial folds', len(y), len(np.unique(folds))
    )
    candidates: list[dict[str, Any]] = []
    alpha = 100.0
    if len(np.unique(folds)) >= 2:
        for a in (100.0, 10.0, 1.0):
            losses: list[float] = []
            for fold in np.unique(folds):
                tr, te = folds != fold, folds == fold
                local = fit_ridge(x[tr], y[tr], a, gpu)
                losses.append(float(np.mean((predict(x[te], local, gpu, clip=False) - y[te]) ** 2)))
            candidates.append(
                {
                    'alpha': a,
                    'fold_mse': losses,
                    'mean_mse': float(np.mean(losses)),
                    'se_mse': float(np.std(losses, ddof=1) / np.sqrt(len(losses))),
                }
            )
        best = min(candidates, key=lambda c: c['mean_mse'])
        alpha = max(
            c['alpha'] for c in candidates if c['mean_mse'] <= best['mean_mse'] + best['se_mse']
        )
    LOGGER.info(
        'Selected Ridge alpha=%g (%s)',
        alpha,
        'one-standard-error CV' if candidates else 'single-block fallback',
    )
    model = fit_ridge(x, y, alpha, gpu)
    model.update(
        protocol=PROTOCOL,
        features=FEATURES,
        delta=DELTA,
        schema='seven_columns_percent_single_base_v1',
        cv=candidates,
        cv_status='tuned_1se' if candidates else 'one_block_fixed_alpha100',
        training_rows=len(y),
        max_train=max_train,
        seed=seed,
        sampling='smallest_numpy_PCG64_uint64_per_contig_sorted_start_then_key_v1',
        quantizer=fit_quantizer(predict(x, model, gpu), blocks),
    )
    return model


def save(path: str, model: dict[str, Any]) -> None:
    """Serialize portable model parameters as strict JSON without NaN values."""
    Path(path).write_text(json.dumps(model, indent=2, allow_nan=False))


def load(path: str) -> dict[str, Any]:
    """Read a frozen model and validate its scientific protocol and feature contract."""
    model = json.loads(Path(path).read_text())
    if (
        model.get('protocol') != PROTOCOL
        or model.get('features') != FEATURES
        or model.get('delta') != DELTA
        or model.get('schema') != 'seven_columns_percent_single_base_v1'
    ):
        raise ValueError('Incompatible frozen model scientific/schema protocol')
    for name in ('fill', 'mean', 'scale', 'coef'):
        a = np.asarray(model[name], dtype=float)
        if a.shape != (len(FEATURES),) or not np.isfinite(a).all():
            raise ValueError(f'Invalid frozen model {name}')
    if np.any(np.asarray(model['scale']) <= 0) or not np.isfinite(model['intercept']):
        raise ValueError('Invalid frozen model scaling/intercept')
    q = model['quantizer']
    edges, values, cuts = (np.asarray(q[k], dtype=float) for k in ('edges', 'values', 'cuts'))
    if (
        not (1 <= len(values) <= 128 and len(edges) + 1 == len(values))
        or not np.isfinite(np.r_[edges, values, cuts]).all()
        or np.any(np.diff(edges) <= 0)
        or np.any(np.diff(values) <= 0)
        or np.any((values < 0) | (values > 1))
        or (q['supported'] and (len(cuts) != 3 or np.any(np.diff(cuts) <= 0)))
    ):
        raise ValueError('Invalid frozen quantizer')
    return model
