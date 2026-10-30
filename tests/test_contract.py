"""Fixed small fixtures; no raw research datasets or external network."""

from __future__ import annotations

import gzip
import importlib.util
import io
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from methridgeqc import _model, _science, qc
from methridgeqc import eval as ev
from methridgeqc._io import Reference, batches

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('METHRIDGEQC_PYTHON', os.sys.executable)


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = Path(tempfile.mkdtemp(prefix='test-', dir=ROOT))
        self.reference = self.folder / 'ref.fa'
        self.reference.write_text('>chrTiny\nACGTCGCGAA\n>alt\nCGCGCGCGCG\n')
        self.frame = pl.DataFrame(
            {
                'chrom': ['chrTiny'] * 7 + ['alt'] * 8,
                'start': [1, 2, 4, 6, 7, 0, 5] + list(range(8)),
                'end': [2, 3, 5, 7, 8, 1, 6] + list(range(1, 9)),
                'm': [8.0, 2.0, 0.0, 1.0, 1.5, 2.0, 2.0] + [float(i) for i in range(8)],
                'u': [2.0, 8.0, 0.0, 9.0, 1.0, 2.0, 2.0] + [float(10 - i) for i in range(8)],
                'depth': [10.0, 10.0, 0.0, 10.0, 2.5, 4.0, 4.0] + [10.0] * 8,
                'beta': [80.0, 20.0, 0.0, 10.0, 60.0, 50.0, 50.0]
                + [float(i * 10) for i in range(8)],
                'tag': ['preserve'] * 15,
            }
        )
        self.source = self.folder / 'input.tsv'
        self.frame.write_csv(self.source, separator='\t')
        self.truth = self.folder / 'truth.tsv'
        pl.DataFrame(
            {
                'chrom': ['chrTiny'] * 4 + ['alt'] * 8,
                'start': [1, 2, 4, 6] + list(range(8)),
                'end': [2, 3, 5, 7] + list(range(1, 9)),
                'strand': ['+', '-', '+', '+'] + ['+', '-'] * 4,
                'type': ['CG'] * 12,
                'beta': [80.0, 90.0, 101.0, 0.0] + [float(i * 10) for i in range(8)],
            }
        ).write_csv(self.truth, separator='\t')

    def tearDown(self) -> None:
        shutil.rmtree(self.folder)

    def run_qc(self, name: str, **kwargs: object) -> Path:
        output = self.folder / name
        qc.run(
            str(self.source),
            str(self.reference),
            str(output),
            work_dir=str(self.folder / 'work'),
            **kwargs,
        )
        return output

    def test_flags_and_formats(self) -> None:
        for score, classify in [(False, False), (True, False), (False, True), (True, True)]:
            out = self.run_qc(
                f'flags-{score}-{classify}.parquet',
                output_format='parquet.lz4',
                score=score,
                classify=classify,
            )
            frame = pl.read_parquet(out)
            self.assertEqual('qc_risk' in frame.columns, score)
            self.assertEqual('qc_grade' in frame.columns, classify)
            self.assertEqual(len(frame), 14)
            self.assertEqual(set(frame['chrom']), {'alt', 'chrTiny'})
            self.assertEqual(
                frame.filter(pl.col('start') == 7).filter(pl.col('chrom') == 'chrTiny')[
                    'qc_reason'
                ][0],
                'invalid_counts',
            )
            self.assertEqual(pq.ParquetFile(out).metadata.row_group(0).column(0).compression, 'LZ4')
            self.assertEqual(pq.ParquetFile(out).schema_arrow.metadata[b'beta_unit'], b'percent')
        for fmt in ('bedgraph', 'bedgraph.gz', 'parquet.lz4'):
            out = self.run_qc('roundtrip.' + fmt, output_format=fmt, score=True, classify=True)
            restored = pl.concat(list(batches(str(out))))
            original = self.frame.filter(
                ~((pl.col('chrom') == 'chrTiny') & (pl.col('start') == 0))
            ).sort('chrom', 'start')
            self.assertEqual(
                restored.sort('chrom', 'start').select(original.columns).to_dicts(),
                original.to_dicts(),
            )
            report = self.folder / ('report-' + fmt + '.tsv')
            ev.run(
                str(out),
                str(self.reference),
                str(self.truth),
                str(report),
                bootstrap=4,
                work_dir=str(self.folder / 'work'),
            )
            self.assertIn(
                'invalid_record', ' '.join(pl.read_csv(report, separator='\t')['metric'].to_list())
            )

    def test_input_order_model_reuse(self) -> None:
        model = self.folder / 'model.json'
        a = self.run_qc(
            'a.parquet',
            output_format='parquet.lz4',
            score=True,
            classify=True,
            model_out=str(model),
            max_train=5,
        )
        parquet = self.folder / 'shuffled.parquet'
        self.frame.reverse().write_parquet(parquet)
        b = self.folder / 'b.parquet'
        qc.run(
            str(parquet),
            str(self.reference),
            str(b),
            output_format='parquet.lz4',
            score=True,
            classify=True,
            max_train=5,
            threads=2,
            work_dir=str(self.folder / 'work'),
        )
        self.assertEqual(pl.read_parquet(a).to_dicts(), pl.read_parquet(b).to_dicts())
        c = self.run_qc(
            'c.parquet', output_format='parquet.lz4', score=True, classify=True, model_in=str(model)
        )
        self.assertEqual(pl.read_parquet(a).to_dicts(), pl.read_parquet(c).to_dicts())
        env = dict(os.environ, PYTHONPATH=str(Path(qc.__file__).resolve().parents[1]))
        result = subprocess.run(
            [
                PYTHON,
                '-m',
                'methridgeqc.cli',
                'qc',
                '--input',
                '-',
                '--reference',
                str(self.reference),
                '--output',
                '-',
                '--output-format',
                'bedgraph.gz',
                '--score',
                '--classify',
                '--max-train',
                '5',
                '--work-dir',
                str(self.folder / 'work'),
            ],
            input=self.frame.reverse().write_csv(separator='\t').encode(),
            capture_output=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(gzip.decompress(result.stdout).startswith(b'#{'))
        stdin = self.folder / 'stdin.bedgraph.gz'
        stdin.write_bytes(result.stdout)
        got = pl.concat(list(batches(str(stdin))))
        self.assertEqual(got.to_dicts(), pl.read_parquet(a).select(got.columns).to_dicts())
        result = subprocess.run(
            [
                PYTHON,
                '-m',
                'methridgeqc.cli',
                'qc',
                '--input',
                '-',
                '--input-format',
                'parquet',
                '--reference',
                str(self.reference),
                '--output',
                str(self.folder / 'stdin.parquet'),
                '--output-format',
                'parquet.lz4',
                '--score',
                '--classify',
                '--max-train',
                '5',
                '--work-dir',
                str(self.folder / 'work'),
            ],
            input=parquet.read_bytes(),
            capture_output=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            pl.read_parquet(a).to_dicts(), pl.read_parquet(self.folder / 'stdin.parquet').to_dicts()
        )

    def test_validation_failures(self) -> None:
        for label, frame in [
            ('duplicate', pl.concat([self.frame, self.frame.slice(0, 1)])),
            ('unknown', self.frame.with_columns(pl.lit('unknown').alias('chrom'))),
            ('interval', self.frame.with_columns((pl.col('end') + 1).alias('end'))),
            ('negative_coordinate', self.frame.with_columns(pl.lit(-1).alias('start'))),
            ('beta_fraction_not_auto', self.frame.with_columns(pl.lit(101.0).alias('beta'))),
        ]:
            source = self.folder / (label + '.parquet')
            frame.write_parquet(source)
            with self.assertRaises(ValueError):
                qc.run(
                    str(source),
                    str(self.reference),
                    str(self.folder / 'unused'),
                    work_dir=str(self.folder / 'work'),
                )
        conflict = self.frame.with_columns(pl.lit('U').alias('qc_grade'))
        conflict.write_csv(self.source, separator='\t')
        with self.assertRaises(ValueError):
            self.run_qc('unused')

    def test_truth_status_and_percent(self) -> None:
        out = self.run_qc('truth.parquet', output_format='parquet.lz4', score=True, classify=True)
        report = self.folder / 'eval.tsv'
        ev.run(
            str(out),
            str(self.reference),
            str(self.truth),
            str(report),
            bootstrap=3,
            work_dir=str(self.folder / 'work'),
        )
        data = pl.read_csv(report, separator='\t', null_values='NA')
        valid = data.filter((pl.col('group') == 'all') & (pl.col('metric') == 'truth_valid'))[
            'estimate'
        ][0]
        self.assertEqual(valid, 11.0)
        t = ev.truth_frames(
            str(self.truth), None, Reference(self.reference), 'percent', self.folder, None
        )
        self.assertEqual(t['chrTiny']['truth'][0], 0.8)
        self.assertEqual(t['chrTiny']['truth_status'][2], 'invalid_record')
        duplicate = pl.read_csv(self.truth, separator='\t')
        pl.concat([duplicate, duplicate.slice(0, 1)]).write_csv(self.truth, separator='\t')
        with self.assertRaises(ValueError):
            ev.truth_frames(
                str(self.truth), None, Reference(self.reference), 'percent', self.folder, None
            )

    def test_null_truth_and_full_coverage(self) -> None:
        source = self.folder / 'small.parquet'
        self.frame.filter((pl.col('chrom') == 'alt') & (pl.col('start') < 6)).write_parquet(source)
        truth = self.folder / 'null-truth.parquet'
        pl.DataFrame(
            {
                'chrom': ['alt'] * 5,
                'start': list(range(5)),
                'end': list(range(1, 6)),
                'strand': ['+', '-', None, '-', '+'],
                'type': ['CG', None, 'CG', 'CG', 'CG'],
                'beta': [0.0, 10.0, 20.0, None, 101.0],
            }
        ).write_parquet(truth)
        report = self.folder / 'null-report.tsv'
        ev.run(
            str(source),
            str(self.reference),
            str(truth),
            str(report),
            bootstrap=0,
            work_dir=str(self.folder / 'work'),
        )
        data = pl.read_csv(report, separator='\t', null_values='NA').filter(
            pl.col('group') == 'all'
        )

        def metric(name: str) -> object:
            return data.filter(pl.col('metric') == name)['estimate'][0]

        self.assertEqual(metric('coverage'), 1.0)
        self.assertEqual(metric('truth_valid'), 1.0)
        self.assertEqual(metric('truth_invalid_record'), 4.0)
        self.assertEqual(metric('truth_absent'), 1.0)
        self.assertEqual(metric('RMSE'), 0.0)
        self.assertEqual(data.filter(pl.col('metric') == 'coverage')['n'][0], 6.0)
        self.assertEqual(data.filter(pl.col('metric') == 'RMSE')['n'][0], 1.0)
        for mode in ('invalid', 'absent'):
            original = pl.read_parquet(truth)
            (
                original.with_columns(pl.lit(None, dtype=pl.Float64).alias('beta'))
                if mode == 'invalid'
                else original.head(0)
            ).write_parquet(truth)
            ev.run(
                str(source),
                str(self.reference),
                str(truth),
                str(report),
                bootstrap=0,
                work_dir=str(self.folder / 'work'),
            )
            data = pl.read_csv(report, separator='\t', null_values='NA').filter(
                pl.col('group') == 'all'
            )
            self.assertIsNone(metric('RMSE'))
            self.assertEqual(metric('coverage'), 1.0)
            self.assertEqual(metric('truth_absent'), 6.0 if mode == 'absent' else 1.0)
            self.assertEqual(data.filter(pl.col('metric') == 'RMSE')['n'][0], 0.0)

    def test_ties_bootstrap_missing(self) -> None:
        s, diag = ev.expected_retention(np.array([0.1, 0.1, 0.3]), np.ones(3), 0.25, 3.0)
        np.testing.assert_equal(s, [0.375, 0.375, 0.0])
        self.assertEqual(diag['achieved_coverage'], 0.25)
        frame = pl.DataFrame(
            {
                'chrom': ['alt'] * 3,
                '_block': [0] * 3,
                'error': [0.0, 0.2, 0.3],
                'truth_status': ['valid'] * 3,
                'qc_risk_quantized': [0.1, 0.1, 0.3],
            }
        )
        a = ev.evaluate(frame, {'alt': 1}, 8)
        b = ev.evaluate(frame, {'alt': 1}, 8)
        self.assertEqual(a.to_dicts(), b.to_dicts())
        constant = frame.with_columns(pl.lit(0.0).alias('error'))
        report = ev.evaluate(constant, {'alt': 1}, 2)
        self.assertEqual(
            set(report.filter(pl.col('metric') == 'AUROC')['status']), {'single_class'}
        )
        self.assertEqual(
            report.filter(pl.col('group') == 'grade')['status'][0], 'missing_grade_column'
        )
        no_score = ev.evaluate(frame.drop('qc_risk_quantized'), {'alt': 1}, 2)
        self.assertEqual(
            no_score.filter(pl.col('group') == 'score')['status'][0], 'missing_score_columns'
        )

    def test_science_fixed_regression(self) -> None:
        f = _science.features(
            {
                'm': np.array([8.0, 2.0, 1.0]),
                'u': np.array([2.0, 8.0, 9.0]),
                'depth': np.array([10.0, 10.0, 10.0]),
                'om': np.array([2.0, 8.0, np.nan]),
                'opposite_depth': np.array([10.0, 10.0, np.nan]),
                'input_legal': np.ones(3),
                'strand': np.array(['+', '-', '+']),
            },
            0.10,
        )
        self.assertAlmostEqual(f['p_raw'][0], 0.8)
        self.assertAlmostEqual(f['posterior_sd'][0], np.sqrt(8.5 * 2.5 / (11**2 * 12)))
        self.assertEqual(f['r_anchor'][0], f['r_strand'][0])
        self.assertEqual(f['r_anchor'][2], f['r_count'][2])
        # Explicit opt-in read-only source regression; production package has no parent imports.
        parent = ROOT.parent
        core_path = parent / 'unsupervised/qc_core.py'
        if not core_path.exists():
            return
        spec = importlib.util.spec_from_file_location('research_core_test', core_path)
        assert spec is not None and spec.loader is not None
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        source = {
            k: v
            for k, v in f.items()
            if k in ('m', 'u', 'depth', 'om', 'opposite_depth', 'input_legal', 'strand')
        }
        expected = old.features(source, 0.10)
        for key in _science.FEATURES + ['r_count', 'r_strand', 'r_anchor']:
            np.testing.assert_allclose(f[key], expected[key], rtol=0, atol=0, equal_nan=True)
        spec = importlib.util.spec_from_file_location(
            'research_quantizer_test', parent / 'alternative_models/quantization.py'
        )
        assert spec is not None and spec.loader is not None
        oldq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oldq)
        risks = np.linspace(0, 1, 512)
        blocks = np.arange(512) % 8
        self.assertEqual(_science.fit_quantizer(risks, blocks), oldq.fit_quantizer(risks, blocks))

    def test_cv_fold_preprocessing(self) -> None:
        x = np.tile(np.arange(16.0)[:, None], (1, 8))
        x[::2, 3] = np.nan
        y = np.linspace(0.1, 0.9, 16)
        blocks = np.asarray([f'chr\0{i // 4}' for i in range(16)])
        model = _model.train(x, y, blocks, False, 200000, 1729)
        self.assertEqual(len(model['cv']), 3)
        self.assertEqual(len(model['cv'][0]['fold_mse']), 4)
        local = _model.fit_ridge(x[:4], y[:4], 100.0, False)
        self.assertEqual(local['fill'][0], 1.5)
        self.assertNotEqual(local['fill'][0], model['fill'][0])
        best = min(model['cv'], key=lambda r: r['mean_mse'])
        eligible = [
            r['alpha'] for r in model['cv'] if r['mean_mse'] <= best['mean_mse'] + best['se_mse']
        ]
        self.assertEqual(model['alpha'], max(eligible))
        one = _model.train(x[:4], y[:4], np.full(4, 'one'), False, 200000, 1729)
        self.assertEqual(one['cv_status'], 'one_block_fixed_alpha100')
        self.assertFalse(one['quantizer']['supported'])

    def test_empty_invalid_and_resource_bounds(self) -> None:
        self.frame.filter((pl.col('chrom') == 'chrTiny') & (pl.col('start') == 0)).write_csv(
            self.source, separator='\t'
        )
        out = self.run_qc('empty.parquet', output_format='parquet.lz4', score=True, classify=True)
        empty = pl.read_parquet(out)
        self.assertEqual(len(empty), 0)
        self.assertIn('qc_grade', empty.columns)
        self.frame.with_columns(
            pl.lit(0.0).alias('m'), pl.lit(0.0).alias('u'), pl.lit(0.0).alias('depth')
        ).write_csv(self.source, separator='\t')
        invalid = pl.read_parquet(
            self.run_qc('invalid.parquet', output_format='parquet.lz4', score=True, classify=True)
        )
        self.assertTrue(invalid['qc_risk'].is_null().all())
        self.assertEqual(set(invalid['qc_grade']), {'U'})
        for kwargs in ({'threads': 121}, {'max_train': 200001}, {'max_train': 0}):
            with self.assertRaises(ValueError):
                self.run_qc('bounds', **kwargs)
        self.assertFalse((self.folder / 'work').exists())

    def test_cli_eval_default_bootstrap(self) -> None:
        env = dict(os.environ, PYTHONPATH=str(Path(qc.__file__).resolve().parents[1]))
        out = self.folder / 'cli.parquet'
        common = ['--reference', str(self.reference), '--work-dir', str(self.folder / 'work')]
        result = subprocess.run(
            [
                PYTHON,
                '-m',
                'methridgeqc.cli',
                'qc',
                '--input',
                str(self.source),
                '--output',
                str(out),
                '--output-format',
                'parquet.lz4',
                '--score',
                '--classify',
            ]
            + common,
            capture_output=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            [
                PYTHON,
                '-m',
                'methridgeqc.cli',
                'eval',
                '--input',
                str(out),
                '--truth',
                str(self.truth),
                '--output',
                '-',
            ]
            + common,
            capture_output=True,
            env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = pl.read_csv(io.BytesIO(result.stdout), separator='\t', null_values='NA')
        self.assertIn('quantized', set(report['group']))
        self.assertIn(b'"bootstrap": 300', result.stderr)
        self.assertNotIn(b'Failure evidence', result.stdout)

    def test_gpu_contract(self) -> None:
        if os.environ.get('METHRIDGEQC_TEST_GPU') != '1':
            with patch.dict('sys.modules', {'cupy': None, 'cuml': None}):
                with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                    _model.gpu_modules()
            return
        _model.gpu_modules()
        rng = np.random.default_rng(1729)
        x = rng.normal(size=(40, 8))
        x[0, 3] = np.nan
        y = np.clip(rng.normal(0.5, 0.1, 40), 0, 1)
        cpu = _model.fit_ridge(x, y, 10.0, False)
        gpu = _model.fit_ridge(x, y, 10.0, True)
        np.testing.assert_allclose(
            _model.predict(x, cpu), _model.predict(x, gpu, True), rtol=1e-7, atol=1e-8
        )
        out = self.run_qc(
            'gpu.parquet', output_format='parquet.lz4', score=True, classify=True, gpu=True
        )
        self.assertEqual(len(pl.read_parquet(out)), 14)


if __name__ == '__main__':
    unittest.main()
