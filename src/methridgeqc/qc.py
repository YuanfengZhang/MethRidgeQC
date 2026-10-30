"""Public chromosome-wise count QC; never changes original measurements."""

from __future__ import annotations

import json
import logging
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl
from threadpoolctl import threadpool_limits

from . import _model
from ._io import (
    OUTPUT_FORMATS,
    RAW,
    InputFormat,
    Output,
    OutputFormat,
    Reference,
    Workspace,
    legal_counts,
    numeric,
    stage,
    validate_integer,
)
from ._science import DELTA, FEATURES, apply_quantizer, features

LOGGER: logging.Logger = logging.getLogger(__name__)


def chromosome_features(frame: pl.DataFrame, anchors: bool = True) -> dict[str, npt.NDArray[Any]]:
    """Pair observed opposite strands without merging cytosines or inventing counts."""
    starts = frame['start'].cast(pl.Int64).to_numpy()
    strand = frame['strand'].to_numpy()
    lookup = {int(s): i for i, s in enumerate(starts)}
    opposite = np.asarray(
        [lookup.get(int(s + (1 if t == '+' else -1)), -1) for s, t in zip(starts, strand)],
        dtype=np.int64,
    )
    legal = legal_counts(frame)
    indices = np.maximum(opposite, 0)
    observed = (opposite >= 0) & legal[indices]
    data = {k: numeric(frame, k) for k in ('m', 'u', 'depth')}
    data.update(
        strand=strand,
        input_legal=legal.astype(int),
        opposite_depth=np.where(observed, data['depth'][indices], np.nan),
        om=np.where(observed, data['m'][indices], np.nan),
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        return features(data, DELTA, anchors)


def sample_chromosome(
    item: tuple[str, Path], cap: int, seed: int
) -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.str_],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.str_],
]:
    """Select capped stable training keys and compute anchors only for selected rows."""
    chrom, path = item
    frame = pl.read_parquet(path).sort('start')
    # Anchors only for preregistered training rows, not all inference rows.
    f = chromosome_features(frame, anchors=False)
    valid = np.flatnonzero(f['valid'])
    keys = _model.stable_keys(chrom, frame['start'].to_numpy()[valid], seed)
    order = np.lexsort((frame['start'].to_numpy()[valid], keys))[:cap]
    selected = valid[order]
    subset = {
        k: v[selected]
        for k, v in f.items()
        if k in ('m', 'u', 'depth', 'strand', 'input_legal', 'opposite_depth', 'om')
    }
    anchor = features(subset, DELTA)
    x = np.column_stack([anchor[k] for k in FEATURES])
    blocks = np.asarray(
        [
            f'{chrom}\0{int(v) // 1_000_000:012d}'
            for v in frame['canonical_start'].to_numpy()[selected]
        ]
    )
    identities = np.asarray(
        [f'{chrom}\0{int(v):020d}' for v in frame['start'].to_numpy()[selected]]
    )
    return keys[order], identities, x, anchor['r_anchor'], blocks


def score_chromosome(
    item: tuple[str, Path], model: dict[str, Any] | None, score: bool, classify: bool, gpu: bool
) -> pl.DataFrame:
    """Score every eligible row of one chromosome using a shared frozen model."""
    chrom, path = item
    frame = pl.read_parquet(path).sort('start')
    LOGGER.debug('Processing chromosome %s (%d rows)', chrom, len(frame))
    if model is None:
        return frame
    f = chromosome_features(frame, anchors=False)
    valid = f['valid']
    risk = np.full(len(frame), np.nan)
    quantized = np.full(len(frame), np.nan)
    grades = np.full(len(frame), 'U', dtype='<U2')
    if valid.any():
        x = np.column_stack([f[k][valid] for k in FEATURES])
        risk[valid] = _model.predict(x, model, gpu)
        quantized[valid], grades[valid] = apply_quantizer(risk[valid], model['quantizer'])
    if score:
        frame = frame.with_columns(
            pl.Series('qc_risk', risk, nan_to_null=True),
            pl.Series('qc_risk_quantized', quantized, nan_to_null=True),
        )
    if classify:
        frame = frame.with_columns(pl.Series('qc_grade', grades))
        if not model['quantizer']['supported']:
            frame = frame.with_columns(
                pl.when(pl.col('qc_valid'))
                .then(pl.lit('grade_unsupported:' + model['quantizer']['reason']))
                .otherwise(pl.col('qc_reason'))
                .alias('qc_reason')
            )
    return frame


def run(
    input: str,
    reference: str,
    output: str,
    *,
    input_format: InputFormat | None = None,
    output_format: OutputFormat = 'bedgraph',
    score: bool = False,
    classify: bool = False,
    gpu: bool = False,
    threads: int | None = 1,
    work_dir: str | None = None,
    model_out: str | None = None,
    model_in: str | None = None,
    max_train: int = 200000,
    seed: int = 1729,
    chromosomes: list[str] | None = None,
) -> dict[str, Any]:
    """Validate CpG counts and optionally score/classify whole chromosomes.

    Parameters
    ----------
    input, reference, output : str
        Count table, uncompressed FASTA and destination; ``-`` selects stdin/stdout.
    score, classify : bool
        Independently export risk columns and reliability grades. Neither flag
        changes the original counts or beta values.
    threads : int or None
        Chromosome workers in 1..120; None requests automatic numerical threads.
    model_in, model_out : str or None
        Reuse a compatible frozen model or save a newly fitted JSON model.
    max_train : int
        Deterministic training cap in 1..200000; never limits scored rows.

    Returns
    -------
    dict[str, Any]
        Run statistics, also emitted as a machine-readable JSON line on stderr.

    Raises
    ------
    ValueError
        An input contract or resource parameter is invalid.
    RuntimeError
        Explicit GPU execution is unavailable. Failed workspaces are retained.
    """
    if threads is not None:
        threads = validate_integer(threads, 'threads', 1, 120)
    max_train = validate_integer(max_train, 'max_train', 1, 200000)
    seed = validate_integer(seed, 'seed')
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(
            f'Unsupported output format {output_format!r}; choose from {OUTPUT_FORMATS}'
        )
    LOGGER.info(
        'Starting QC: input=%s output=%s backend=%s', input, output, 'GPU' if gpu else 'CPU'
    )
    if gpu:
        _model.gpu_modules()
    ref = Reference(reference)
    work = Workspace(work_dir)
    LOGGER.debug('Temporary workspace: %s', work.path)
    try:
        paths, info = stage(input, input_format, ref, work.path, chromosomes)
        LOGGER.info(
            'Validated %d input rows across %d chromosomes; filtered %d non-CpG rows',
            info['input_rows'],
            len(paths),
            info['non_cpg'],
        )
        if info['invalid_counts'] or info['zero_depth']:
            LOGGER.warning(
                'Retaining %d invalid-count and %d zero-depth rows without risk scores',
                info['invalid_counts'],
                info['zero_depth'],
            )
        model: dict[str, Any] | None = None
        workers = 1 if gpu or threads is None else threads
        limits = nullcontext() if threads is None else threadpool_limits(limits=1)
        with limits, ThreadPoolExecutor(max_workers=workers) as pool:
            if score or classify:
                if model_in:
                    LOGGER.info('Loading frozen model: %s (no refitting)', model_in)
                    model = _model.load(model_in)
                else:
                    sample: tuple[npt.NDArray[Any], ...] | None = None
                    items = iter(paths.items())
                    pending: list[Future[Any]] = []
                    for _ in range(workers):
                        item = next(items, None)
                        if item is not None:
                            pending.append(pool.submit(sample_chromosome, item, max_train, seed))
                    while pending:
                        current = pending.pop(0).result()
                        merged = (
                            current
                            if sample is None
                            else tuple(np.concatenate([a, b]) for a, b in zip(sample, current))
                        )
                        selected = np.lexsort((merged[1], merged[0]))[:max_train]
                        sample = tuple(a[selected] for a in merged)
                        item = next(items, None)
                        if item is not None:
                            pending.append(pool.submit(sample_chromosome, item, max_train, seed))
                    if sample is None or not len(sample[0]):
                        info['training_status'] = 'no_valid_training_records'
                        LOGGER.warning(
                            'No valid training records; requested scores remain missing and grades remain U'
                        )
                    else:
                        _, _, x, y, blocks = sample
                        model = _model.train(x, y, blocks, gpu, max_train, seed)
                        if model_out:
                            _model.save(model_out, model)
                            LOGGER.info('Saved frozen model: %s', model_out)
                if model is not None and classify and not model['quantizer']['supported']:
                    LOGGER.warning(
                        'Reliability grading unsupported (%s); retaining U grades',
                        model['quantizer'].get('reason', 'insufficient spatial support'),
                    )
            sink = Output(output, output_format)
            try:
                # Bounded ordered submission: no pool.map prefetch of all chromosomes.
                items = iter(paths.items())
                pending: list[Future[Any]] = []
                for _ in range(workers):
                    item = next(items, None)
                    if item is not None:
                        pending.append(
                            pool.submit(score_chromosome, item, model, score, classify, gpu)
                        )
                while pending:
                    frame = pending.pop(0).result()
                    if model is None and (score or classify):
                        if score:
                            frame = frame.with_columns(
                                pl.lit(None, dtype=pl.Float64).alias('qc_risk'),
                                pl.lit(None, dtype=pl.Float64).alias('qc_risk_quantized'),
                            )
                        if classify:
                            frame = frame.with_columns(pl.lit('U').alias('qc_grade'))
                    sink.write(frame)
                    item = next(items, None)
                    if item is not None:
                        pending.append(
                            pool.submit(score_chromosome, item, model, score, classify, gpu)
                        )
                empty = pl.DataFrame(
                    schema={k: pl.String if k == 'chrom' else pl.Float64 for k in RAW}
                )
                sink.close(empty)
            except BaseException:
                sink.close()
                raise
        info.update(
            chromosomes=list(paths),
            backend=model['backend'] if model else None,
            training_rows=model['training_rows'] if model else 0,
            max_train=max_train,
            seed=seed,
            model_reused=bool(model_in),
            cv_status=model['cv_status'] if model else None,
            quantizer_supported=model['quantizer']['supported'] if model else None,
        )
        LOGGER.info(
            'QC complete: %d chromosomes, %d training records; output=%s',
            len(paths),
            info['training_rows'],
            output,
        )
        print(json.dumps(info), file=sys.stderr)
        work.finish()
        return info
    except BaseException:
        LOGGER.error('Failure evidence retained: %s', work.path)
        raise
