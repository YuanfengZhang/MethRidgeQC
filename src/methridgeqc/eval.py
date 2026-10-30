"""Public frozen-QC evaluation; expected ties and chromosome block bootstrap."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow.parquet as pq

from ._grades import ALL_GRADES, ORDINAL, normalize_frame, normalize_grades
from ._io import (
    InputFormat,
    Reference,
    Workspace,
    batches,
    legal_counts,
    numeric,
    stage,
    validate_integer,
)

type TruthScale = Literal['percent', 'fraction']

LOGGER: logging.Logger = logging.getLogger(__name__)
THRESHOLDS: tuple[float, ...] = (0.05, 0.10, 0.20)


def truth_frames(
    path: str,
    fmt: InputFormat | None,
    reference: Reference,
    scale: TruthScale,
    work: Path,
    chromosomes: list[str] | None,
    *,
    stage_files: bool = False,
) -> dict[str, Any]:
    """Validate truth units and keys, retaining valid/invalid states per chromosome."""
    if scale not in ('percent', 'fraction'):
        raise ValueError('Truth scale must be percent or fraction')
    parts: dict[str, list[pl.DataFrame]] = {}
    writers: dict[str, pq.ParquetWriter] = {}
    paths: dict[str, Path] = {}
    keys = sqlite3.connect(work / 'truth-keys.sqlite')
    keys.execute('PRAGMA temp_store=MEMORY')
    keys.execute(
        'CREATE TABLE IF NOT EXISTS keys(chrom TEXT,start INTEGER,strand TEXT,PRIMARY KEY(chrom,start,strand)) WITHOUT ROWID'
    )
    keys.execute('DELETE FROM keys')
    try:
        for frame in batches(path, fmt, work):
            required = {'chrom', 'start', 'end', 'strand', 'type', 'beta'}
            if not required <= set(frame.columns):
                raise ValueError('Missing six-column truth schema')
            # Batch conversion, never per-row division of percent truth.
            numeric_beta = frame.schema['beta'].is_numeric()
            frame = frame.with_columns(
                (
                    pl.col('beta').cast(pl.Float64, strict=False)
                    / (100.0 if scale == 'percent' else 1.0)
                ).alias('truth')
            )
            for key, group in frame.partition_by('chrom', as_dict=True).items():
                chrom = key[0]
                if chrom not in reference.index:
                    raise ValueError(f'Unknown truth contig: {chrom}')
                if chromosomes is not None and chrom not in chromosomes:
                    continue
                expected = reference.strands(
                    chrom, group['start'].to_numpy(), group['end'].to_numpy()
                )
                observed = group['strand'].fill_null('?').to_numpy()
                kind = group['type'].is_in(['CG', 'CpG']).fill_null(False).to_numpy().astype(bool)
                value = group['truth'].to_numpy()
                valid = (
                    numeric_beta
                    & np.isfinite(value)
                    & (value >= 0)
                    & (value <= 1)
                    & (observed == expected)
                    & (expected != '?')
                    & kind
                )
                try:
                    keys.executemany(
                        'INSERT INTO keys VALUES(?,?,?)',
                        (
                            (chrom, int(start), str(strand))
                            for start, strand in zip(group['start'].to_numpy(), expected)
                        ),
                    )
                    keys.commit()
                except sqlite3.IntegrityError as exc:
                    raise ValueError('Duplicate truth key') from exc
                # Non-CpG truth is outside requested type, not a valid match.
                group = group.with_columns(
                    pl.Series('truth_status', np.where(valid, 'valid', 'invalid_record')),
                    pl.Series('truth', np.where(valid, value, np.nan), nan_to_null=True),
                )
                # Match malformed strand records at their reference CpG coordinate,
                # retaining invalid_record rather than misclassifying them as absent.
                selected = group.select(
                    'chrom', 'start', 'strand', 'truth', 'truth_status'
                ).with_columns(pl.col('start').cast(pl.Int64), pl.Series('strand', expected))
                if selected.select('chrom', 'start', 'strand').is_duplicated().any():
                    raise ValueError('Duplicate reference-normalized truth key')
                if stage_files:
                    table = selected.to_arrow()
                    if chrom not in writers:
                        paths[chrom] = work / f'truth-{len(paths):06d}.parquet'
                        writers[chrom] = pq.ParquetWriter(
                            paths[chrom], table.schema, compression='lz4'
                        )
                    writers[chrom].write_table(table)
                else:
                    parts.setdefault(chrom, []).append(selected)
    finally:
        for writer in writers.values():
            writer.close()
        keys.close()
    return (
        paths
        if stage_files
        else {c: pl.concat(v, how='vertical_relaxed') for c, v in parts.items()}
    )


def expected_retention(
    score: npt.NDArray[np.float64], weights: npt.NDArray[np.float64], coverage: float, total: float
) -> tuple[npt.NDArray[np.float64], dict[str, float]]:
    """Allocate expected retention across score ties without using truth to break ties."""
    selected = np.zeros(len(score))
    remaining = coverage * total
    boundary = float('nan')
    take = float('nan')
    for level in np.unique(score[np.isfinite(score)]):
        mask = score == level
        count = weights[mask].sum()
        if count <= 0:
            continue
        take = min(1.0, max(0.0, remaining / count))
        selected[mask] = take
        boundary = float(level)
        remaining -= take * count
        if remaining <= 1e-9:
            break
    return selected, {
        'achieved_coverage': float(np.sum(weights * selected) / total) if total else float('nan'),
        'boundary_risk': boundary,
        'boundary_tie_retained_fraction': take,
        'supported': float(remaining <= max(1e-7, coverage * total * 1e-10)),
    }


def block_weights(
    blocks: list[tuple[str, int]], draws: int, universe: dict[str, int], seed: int = 1729
) -> npt.NDArray[np.float64]:
    """Draw reproducible chromosome-local block weights, including unobserved blocks."""
    weights = np.ones((draws + 1, len(blocks)), dtype=float)
    for chrom in sorted({c for c, _ in blocks}):
        count = universe[chrom]
        indices = [i for i, (c, _) in enumerate(blocks) if c == chrom]
        positions = np.asarray([blocks[i][1] for i in indices], dtype=int)
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, *chrom.encode('utf-8')])
        )
        for draw in range(1, draws + 1):
            sampled = rng.integers(0, count, size=count)
            weights[draw, indices] = np.bincount(sampled, minlength=count)[positions]
    return weights


def evaluate(
    frame: pl.DataFrame, universe: dict[str, int], bootstrap: int = 300, seed: int = 1729
) -> pl.DataFrame:
    """Compute frozen-score metrics and conditional block-bootstrap confidence intervals."""
    bootstrap = validate_integer(bootstrap, 'bootstrap', 0)
    seed = validate_integer(seed, 'seed', 0)
    n = len(frame)
    block_keys = list(zip(frame['chrom'].to_list(), frame['_block'].to_list()))
    blocks = sorted(set(block_keys))
    lookup = {b: i for i, b in enumerate(blocks)}
    indices = np.asarray([lookup[b] for b in block_keys], dtype=int)
    draws = block_weights(blocks, bootstrap, universe, seed)
    errors = frame['error'].to_numpy() if 'error' in frame.columns else None
    if errors is not None:
        finite_error = np.isfinite(errors)
        count = np.ones(n)
        matched_count = finite_error.astype(float)
        absolute = np.where(finite_error, np.abs(errors), 0.0)
        squared = np.where(finite_error, errors**2, 0.0)
        events_by_threshold = {
            t: (finite_error & (np.abs(errors) > t)).astype(float) for t in THRESHOLDS
        }
    else:
        count = frame['_n'].to_numpy()
        matched_count = frame['_matched'].to_numpy()
        absolute = frame['_abs'].to_numpy()
        squared = frame['_sq'].to_numpy()
        events_by_threshold = {t: frame[f'_event_{t}'].to_numpy() for t in THRESHOLDS}
    status = frame['truth_status'].to_numpy()
    grades = normalize_grades(frame['qc_grade'].to_list()) if 'qc_grade' in frame.columns else None
    scores: dict[str, npt.NDArray[np.float64]] = {}
    if 'qc_risk_quantized' in frame.columns:
        scores['quantized'] = frame['qc_risk_quantized'].to_numpy()
    elif 'qc_risk' in frame.columns:
        scores['continuous'] = frame['qc_risk'].to_numpy()
    if grades is not None:
        scores['ordinal'] = np.asarray([ORDINAL[g] for g in grades])
    records: dict[tuple[str, float | None, float | None, str], list[float]] = {}
    counts: dict[tuple[str, float | None, float | None, str], float] = {}
    reasons: dict[tuple[str, float | None, float | None, str], str] = {}
    for draw in range(bootstrap + 1):
        weight = draws[draw, indices]
        total = np.sum(weight * count)

        def add(
            group: str,
            metric: str,
            value: float,
            size: float,
            threshold: float | None = None,
            coverage: float | None = None,
            reason: str = 'no_valid_truth_matches',
        ) -> None:
            """Append a weighted metric estimate while retaining support counts and NA reasons."""
            key = (group, threshold, coverage, metric)
            records.setdefault(key, []).append(value)
            if draw == 0:
                counts[key] = size
                reasons[key] = 'ok' if np.isfinite(value) else reason

        groups: dict[str, npt.NDArray[Any]] = {'all': np.ones(n)}
        if grades is not None:
            groups.update({g: (grades == g).astype(float) for g in ALL_GRADES})
            groups['H+MH'] = np.isin(grades, ['H', 'MH']).astype(float)
        for label, selected in groups.items():
            w = weight * selected
            matched = np.sum(w * matched_count)
            size = np.sum(w * count)
            add(label, 'coverage', size / total if total else np.nan, size)
            add(label, 'MAE', float(np.sum(w * absolute) / matched) if matched else np.nan, matched)
            add(
                label,
                'RMSE',
                float(np.sqrt(np.sum(w * squared) / matched)) if matched else np.nan,
                matched,
            )
            for threshold in THRESHOLDS:
                add(
                    label,
                    'event_rate',
                    float(np.sum(w * events_by_threshold[threshold]) / matched)
                    if matched
                    else np.nan,
                    matched,
                    threshold,
                )
            for state in ('valid', 'invalid_record', 'absent'):
                add(
                    label,
                    'truth_' + state,
                    float(np.sum(w[status == state] * count[status == state])),
                    size,
                )
        for kind, risk in scores.items():
            valid = np.isfinite(risk) & (matched_count > 0) & (weight > 0)
            levels, inverse = np.unique(risk[valid], return_inverse=True)
            for threshold in THRESHOLDS:
                positives = np.bincount(
                    inverse,
                    weights=weight[valid] * events_by_threshold[threshold][valid],
                    minlength=len(levels),
                )
                negatives = np.bincount(
                    inverse,
                    weights=weight[valid]
                    * (matched_count[valid] - events_by_threshold[threshold][valid]),
                    minlength=len(levels),
                )
                p, neg = positives.sum(), negatives.sum()
                reason = 'empty_matches' if p + neg == 0 else 'single_class'
                auc = ap = float('nan')
                if p > 0 and neg > 0:
                    auc = float(
                        np.sum(positives * (np.cumsum(negatives) - 0.5 * negatives)) / (p * neg)
                    )
                    pp, nn = np.cumsum(positives[::-1]), np.cumsum(negatives[::-1])
                    ap = float(
                        np.sum(
                            np.divide(pp, pp + nn, out=np.zeros_like(pp), where=pp + nn > 0)
                            * positives[::-1]
                        )
                        / p
                    )
                add(kind, 'AUROC', auc, p + neg, threshold, reason=reason)
                add(kind, 'AUPRC', ap, p + neg, threshold, reason=reason)
            if kind == 'ordinal':
                continue
            for coverage in (0.10, 0.25, 0.50, 0.75):
                selected, diag = expected_retention(risk, weight * count, coverage, total)
                w = selected * weight
                matched = np.sum(w * matched_count)
                supported = bool(diag.pop('supported'))
                for name, value in diag.items():
                    add(kind, name, value, np.sum(w * count), coverage=coverage)
                reason = (
                    'insufficient_scored_coverage' if not supported else 'no_valid_truth_matches'
                )
                add(
                    kind,
                    'MAE',
                    float(np.sum(w * absolute) / matched) if matched and supported else np.nan,
                    matched,
                    coverage=coverage,
                    reason=reason,
                )
                add(
                    kind,
                    'RMSE',
                    float(np.sqrt(np.sum(w * squared) / matched))
                    if matched and supported
                    else np.nan,
                    matched,
                    coverage=coverage,
                    reason=reason,
                )
                for threshold in THRESHOLDS:
                    add(
                        kind,
                        'event_rate',
                        float(np.sum(w * events_by_threshold[threshold]) / matched)
                        if matched and supported
                        else np.nan,
                        matched,
                        threshold,
                        coverage,
                        reason,
                    )
        if not scores:
            add('score', 'AUROC', np.nan, 0.0, reason='missing_score_columns')
        if grades is None:
            add('grade', 'coverage', np.nan, 0.0, reason='missing_grade_column')
    rows: list[dict[str, Any]] = []
    for key, values in records.items():
        group, threshold, coverage, metric = key
        a = np.asarray(values)
        finite = a[1:][np.isfinite(a[1:])]
        lo, hi = (
            np.quantile(finite, [0.025, 0.975])
            if len(finite) and np.isfinite(a[0])
            else (np.nan, np.nan)
        )
        rows.append(
            {
                'domain': 'all_selected_contigs',
                'group': group,
                'threshold': threshold,
                'coverage': coverage,
                'metric': metric,
                'estimate': float(a[0]) if np.isfinite(a[0]) else None,
                'ci_lower': float(lo) if np.isfinite(lo) else None,
                'ci_upper': float(hi) if np.isfinite(hi) else None,
                'n': counts[key],
                'status': reasons[key],
            }
        )
    return pl.DataFrame(rows)


def run(
    input: str,
    reference: str,
    truth: str,
    output: str,
    *,
    input_format: InputFormat | None = None,
    truth_format: InputFormat | None = None,
    truth_scale: TruthScale = 'percent',
    bootstrap: int = 300,
    seed: int = 1729,
    chromosomes: list[str] | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Evaluate existing QC against independent strand-specific methylation truth.

    Parameters
    ----------
    input, reference, truth, output : str
        Frozen QC table, uncompressed FASTA, public truth table and TSV report.
    truth_scale : {'percent', 'fraction'}
        Explicit truth beta units; no automatic unit guessing is performed.
    bootstrap : int
        Non-negative number of conditional 1 Mb block-bootstrap draws.
    seed : int
        Non-negative bootstrap seed for reproducible confidence intervals.

    Returns
    -------
    dict[str, Any]
        Run statistics, also emitted as a JSON line on stderr. No model is fitted
        and no risk or grade columns are predicted when absent from the input.

    Raises
    ------
    ValueError
        Counts, coordinates, truth units or bootstrap parameters are invalid.
    """
    bootstrap = validate_integer(bootstrap, 'bootstrap', 0)
    seed = validate_integer(seed, 'seed', 0)
    if truth_scale not in ('percent', 'fraction'):
        raise ValueError('Truth scale must be percent or fraction')
    LOGGER.info('Starting evaluation: input=%s truth=%s bootstrap=%d', input, truth, bootstrap)
    ref = Reference(reference)
    work = Workspace(work_dir)
    LOGGER.debug('Temporary workspace: %s', work.path)
    try:
        paths, stats = stage(input, input_format, ref, work.path, chromosomes, qc=False)
        truth_by_chrom = truth_frames(
            truth, truth_format, ref, truth_scale, work.path, chromosomes, stage_files=True
        )
        LOGGER.info('Aligning QC and truth across %d chromosomes', len(paths))
        frames: list[pl.DataFrame] = []
        for chrom, path in paths.items():
            LOGGER.debug('Evaluating chromosome %s', chrom)
            frame = pl.read_parquet(path).with_columns(pl.col('start').cast(pl.Int64))
            legal = legal_counts(frame) & (numeric(frame, 'depth') > 0)
            if 'qc_grade' in frame.columns:
                frame = normalize_frame(frame)
                if not frame['qc_grade'].is_in(list(ALL_GRADES)).all():
                    raise ValueError('Invalid QC grade')
                if np.any(~legal & (frame['qc_grade'].to_numpy() != 'U')):
                    raise ValueError('Invalid counts require U')
            for col in ('qc_risk', 'qc_risk_quantized'):
                if col in frame.columns:
                    values = frame[col].cast(pl.Float64).to_numpy()
                    present = ~frame[col].is_null().to_numpy()
                    if np.any(
                        present & (~np.isfinite(values) | (values < 0) | (values > 1))
                    ) or np.any(~legal & present):
                        raise ValueError('Invalid QC risk or risk on invalid counts')
            if chrom in truth_by_chrom:
                frame = frame.join(
                    pl.read_parquet(truth_by_chrom[chrom]),
                    on=['chrom', 'start', 'strand'],
                    how='left',
                )
            else:
                frame = frame.with_columns(
                    pl.lit(None, dtype=pl.Float64).alias('truth'),
                    pl.lit(None, dtype=pl.String).alias('truth_status'),
                )
            frame = frame.with_columns(pl.col('truth_status').fill_null('absent'))
            # Compute errors only on eligible matches; keep every input row for
            # coverage, retention denominators and truth-state accounting.
            eligible = legal & frame['truth_status'].eq('valid').fill_null(False).to_numpy().astype(
                bool
            )
            eligible &= np.isfinite(frame['truth'].to_numpy())
            errors = np.full(len(frame), np.nan)
            matched = frame.filter(pl.Series(eligible))
            errors[eligible] = (
                numeric(matched, 'm') / numeric(matched, 'depth') - matched['truth'].to_numpy()
            )
            frame = frame.with_columns(
                pl.Series('error', errors, nan_to_null=True),
                ((pl.col('start') - (pl.col('strand') == '-').cast(pl.Int64)) // 1_000_000).alias(
                    '_block'
                ),
            )
            score_column = (
                'qc_risk_quantized' if 'qc_risk_quantized' in frame.columns else 'qc_risk'
            )
            categories = [
                c
                for c in ['chrom', '_block', 'truth_status', 'qc_grade', score_column]
                if c in frame.columns
            ]
            frame = frame.with_columns(
                pl.lit(1.0).alias('_n'),
                pl.col('error').is_not_null().cast(pl.Float64).alias('_matched'),
                pl.col('error').abs().fill_null(0.0).alias('_abs'),
                (pl.col('error') ** 2).fill_null(0.0).alias('_sq'),
                *[
                    (pl.col('error').abs() > t)
                    .fill_null(False)
                    .cast(pl.Float64)
                    .alias(f'_event_{t}')
                    for t in THRESHOLDS
                ],
            )
            stats_columns = ['_n', '_matched', '_abs', '_sq'] + [f'_event_{t}' for t in THRESHOLDS]
            frames.append(frame.group_by(categories).agg(pl.col(stats_columns).sum()))
        if frames:
            joined = pl.concat(frames, how='vertical_relaxed')
        else:
            joined = pl.DataFrame(
                schema={
                    'chrom': pl.String,
                    '_block': pl.Int64,
                    'error': pl.Float64,
                    'truth_status': pl.String,
                }
            )
        universe = {c: (ref.index[c][0] + 999999) // 1000000 for c in paths}
        report = evaluate(joined, universe, bootstrap, seed)
        report.write_csv(sys.stdout if output == '-' else output, separator='\t', null_value='NA')
        stats.update(
            truth_scale=truth_scale,
            reference=str(ref.path.resolve()),
            bootstrap=bootstrap,
            seed=seed,
            interval='conditional_on_frozen_model_chromosome_1Mb_blocks',
            chromosomes=list(paths),
        )
        LOGGER.info('Evaluation complete: %d report rows; output=%s', len(report), output)
        print(json.dumps(stats), file=sys.stderr)
        work.finish()
        return stats
    except BaseException:
        LOGGER.error('Failure evidence retained: %s', work.path)
        raise
