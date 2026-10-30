"""Seeded sampling and portable runtime model artifacts."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from methridgeqc import _model, qc
from methridgeqc import eval as evaluation

ROOT = Path(__file__).resolve().parents[1]


def test_sampling_keys_are_reproducible_and_order_independent() -> None:
    starts = np.asarray([10, 2, 30, 4], dtype=np.int64)
    for seed in (1729, -1):
        keys = _model.stable_keys('chr1', starts, seed)
        np.testing.assert_array_equal(keys, _model.stable_keys('chr1', starts, seed))
        order = np.asarray([2, 0, 3, 1])
        np.testing.assert_array_equal(
            keys[order], _model.stable_keys('chr1', starts[order], seed)
        )
        assert keys.dtype == np.uint64
        assert not np.array_equal(keys, _model.stable_keys('chr1', starts, seed + 1))
        assert not np.array_equal(keys, _model.stable_keys('chr2', starts, seed))


def test_bootstrap_is_reproducible_and_contig_local() -> None:
    blocks = [('chr1', 0), ('chr1', 1), ('chr2', 0), ('chr2', 1)]
    universe = {'chr1': 3, 'chr2': 3}
    weights = evaluation.block_weights(blocks, 20, universe)
    np.testing.assert_array_equal(weights, evaluation.block_weights(blocks, 20, universe))
    np.testing.assert_array_equal(
        weights[:, 2:], evaluation.block_weights(blocks[2:], 20, universe)
    )
    np.testing.assert_array_equal(weights[0], np.ones(4))
    assert not np.array_equal(weights, evaluation.block_weights(blocks, 20, universe, 1730))


def test_runtime_model_roundtrip() -> None:
    with tempfile.TemporaryDirectory(prefix='random-test-', dir=ROOT) as folder:
        work = Path(folder)
        output = work / 'qc.parquet.lz4'
        model_path = work / 'ridge.json'
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            qc.run(
                str(ROOT / 'example/BS1_D6_1.parquet.lz4'),
                str(ROOT / 'example/GRCh38_chr22.fa'),
                str(output),
                output_format='parquet.lz4',
                score=True,
                classify=True,
                model_out=str(model_path),
                work_dir=str(work),
            )
            evaluation.run(
                str(output),
                str(ROOT / 'example/GRCh38_chr22.fa'),
                str(ROOT / 'example/D6_reference.parquet.lz4'),
                str(work / 'metrics.tsv'),
                bootstrap=10,
                work_dir=str(work),
            )
        model = json.loads(model_path.read_text())
        assert 'numpy_PCG64' in model['sampling']
        assert _model.load(str(model_path)) == model

        source = pl.read_parquet(ROOT / 'example/BS1_D6_1.parquet.lz4')
        got = pl.read_parquet(output)
        assert got.height > 0
        assert set(got['chrom']) == {'chr22'}
        assert got.select(source.columns).equals(
            source.join(got.select('chrom', 'start'), on=['chrom', 'start']).sort('chrom', 'start')
        )
        assert set(got['qc_grade']) <= {'H', 'MH', 'ML', 'L', 'U'}
        assert np.isfinite(got.filter(pl.col('qc_valid'))['qc_risk'].to_numpy()).all()
        reused = work / 'reused.parquet'
        with contextlib.redirect_stderr(stderr):
            qc.run(
                str(ROOT / 'example/BS1_D6_1.parquet.lz4'),
                str(ROOT / 'example/GRCh38_chr22.fa'), str(reused),
                output_format='parquet.lz4', score=True, classify=True,
                model_in=str(model_path), work_dir=str(work),
            )
        assert got.equals(pl.read_parquet(reused))
        # Compute overall error from coordinates, strands, and valid truth without evaluation helpers.
        truth = pl.read_parquet(ROOT / 'example/D6_reference.parquet.lz4')
        truth = truth.filter(pl.col('type').is_in(['CG', 'CpG']) & pl.col('beta').is_between(0, 100))
        matched = got.filter(pl.col('qc_valid')).join(
            truth.select('chrom', 'start', 'strand', pl.col('beta').alias('truth_beta')),
            on=['chrom', 'start', 'strand'],
        )
        errors = (matched['m'] / matched['depth'] - matched['truth_beta'] / 100).to_numpy()
        assert len(errors) > 0
        report = pl.read_csv(work / 'metrics.tsv', separator='\t', null_values='NA')
        for metric, expected in [('MAE', np.mean(np.abs(errors))), ('RMSE', np.sqrt(np.mean(errors ** 2)))]:
            estimate = report.filter((pl.col('group') == 'all') & (pl.col('metric') == metric))['estimate'][0]
            np.testing.assert_allclose(estimate, expected, rtol=1e-10, atol=1e-12)
