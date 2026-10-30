"""Protect CLI stream contracts, logging isolation and typed public boundaries."""

from __future__ import annotations

import ast
import gzip
import importlib
import inspect
import json
import logging
import os
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from methridgeqc import eval as evaluation
from methridgeqc import qc
from methridgeqc._io import Output, input_format, validate_integer
from methridgeqc.cli import parser

ROOT = Path(__file__).resolve().parents[1]


def cli(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    """Run the current interpreter with the package source and isolated diagnostics."""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    return subprocess.run(
        [sys.executable, '-m', 'methridgeqc.cli', *arguments],
        capture_output=True,
        env=environment,
    )


@pytest.mark.parametrize('value', ['0', '-1', '121', '1.5', 'invalid'])
def test_invalid_thread_arguments(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = [
        'qc',
        '--input',
        'counts',
        '--reference',
        'ref',
        '--output',
        'out',
        '--threads',
        value,
    ]
    with pytest.raises(SystemExit) as failure:
        parser().parse_args(arguments)
    assert failure.value.code == 2
    assert '--threads must be an integer from 1 to 120, or auto' in capsys.readouterr().err


def test_log_level_before_or_after_subcommand() -> None:
    arguments = ['qc', '--input', 'counts', '--reference', 'ref', '--output', 'out']
    assert parser().parse_args(arguments).log_level == 'INFO'
    assert parser().parse_args(['--log-level', 'DEBUG', *arguments]).log_level == 'DEBUG'
    assert parser().parse_args([*arguments, '--log-level', 'ERROR']).log_level == 'ERROR'


def test_cli_import_preserves_lazy_numeric_imports_and_thread_controls() -> None:
    environment = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), OPENBLAS_NUM_THREADS='23')
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import os, sys, logging; import methridgeqc.cli; '
                'assert not logging.getLogger().handlers; '
                "assert os.environ['OPENBLAS_NUM_THREADS'] == '23'; "
                "assert not {'numpy', 'scipy', 'polars', 'cupy', 'cuml'} & sys.modules.keys()"
            ),
        ],
        env=environment,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('level', ['DEBUG', 'ERROR'])
def test_cli_logging_never_contaminates_gzip_stdout(level: str, tmp_path: Path, chr22_subset: Path) -> None:
    result = cli(
        [
            'qc',
            '--input',
            str(chr22_subset),
            '--reference',
            str(ROOT / 'example/GRCh38_chr22.fa'),
            '--output',
            '-',
            '--output-format',
            'bedgraph.gz',
            '--score',
            '--classify',
            '--work-dir',
            str(tmp_path),
            '--log-level',
            level,
        ]
    )
    assert result.returncode == 0, result.stderr
    assert gzip.decompress(result.stdout).startswith(b'#{')
    diagnostics = result.stderr.decode()
    summary = json.loads(diagnostics.splitlines()[-1])
    assert summary['input_rows'] == 256
    if level == 'DEBUG':
        assert '[INFO]' in diagnostics
        assert '[DEBUG]' in diagnostics
    else:
        assert '[INFO]' not in diagnostics
        assert '[DEBUG]' not in diagnostics
        assert '[WARNING]' not in diagnostics


def test_cli_errors_are_actionable_without_default_tracebacks(tmp_path: Path) -> None:
    arguments = [
        'qc',
        '--input',
        str(tmp_path / 'missing'),
        '--reference',
        str(ROOT / 'example/GRCh38_chr22.fa'),
        '--output',
        str(tmp_path / 'output'),
        '--work-dir',
        str(tmp_path / 'work'),
    ]
    normal = cli(arguments)
    debug = cli([*arguments, '--log-level', 'DEBUG'])
    for result in (normal, debug):
        assert result.returncode == 2
        assert not result.stdout
        assert b'Failure evidence retained:' in result.stderr
        assert b'[ERROR]' in result.stderr
    assert b'Traceback' not in normal.stderr
    assert b'Traceback' in debug.stderr


@pytest.mark.parametrize('name', ['threads', 'max_train', 'seed'])
@pytest.mark.parametrize('value', [True, 1.5, '1'])
def test_qc_rejects_invalid_resource_types_before_io(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f'{name} must be an integer'):
        qc.run('missing-input', 'missing-reference', 'unused-output', **{name: value})


@pytest.mark.parametrize('name', ['bootstrap', 'seed'])
@pytest.mark.parametrize('value', [True, 1.5, '1', -1])
def test_eval_rejects_invalid_resource_types_before_io(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=f'{name} must be'):
        evaluation.run(
            'missing-input', 'missing-reference', 'missing-truth', 'unused-output', **{name: value}
        )


def test_format_validation_and_numpy_integer_compatibility(tmp_path: Path) -> None:
    assert type(validate_integer(np.int64(2), 'threads', 1, 120)) is int
    with pytest.raises(ValueError, match='Unsupported input format'):
        input_format('counts.tsv', 'unsupported')
    with pytest.raises(ValueError, match='Unsupported output format'):
        Output(str(tmp_path / 'unused'), 'unsupported')
    assert not (tmp_path / 'unused').exists()
    with pytest.raises(ValueError, match='Unsupported output format'):
        qc.run('missing-input', 'missing-reference', 'unused-output', output_format='unsupported')


def test_api_logs_keep_json_summary_and_measurements(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str], chr22_subset: Path
) -> None:
    output = tmp_path / 'qc.parquet'
    handlers = tuple(logging.getLogger().handlers)
    with caplog.at_level(logging.INFO, logger='methridgeqc'):
        summary = qc.run(
            str(chr22_subset),
            str(ROOT / 'example/GRCh38_chr22.fa'),
            str(output),
            output_format='parquet.lz4',
            score=True,
            classify=True,
            work_dir=str(tmp_path / 'work'),
            threads=np.int64(1),
            max_train=np.int64(200000),
            seed=np.int64(1729),
        )
    assert tuple(logging.getLogger().handlers) == handlers
    assert json.loads(capsys.readouterr().err) == summary
    assert any(record.getMessage().startswith('Starting QC:') for record in caplog.records)
    assert any(
        'Reliability grading unsupported' in record.getMessage() for record in caplog.records
    )
    source = pl.read_parquet(chr22_subset)
    got = pl.read_parquet(output)
    assert got.select(source.columns).equals(source.join(got.select('chrom', 'start'), on=['chrom', 'start']).sort('chrom', 'start'))


def test_source_annotations_and_function_documentation() -> None:
    for directory in ('src/methridgeqc',):
        for source in (ROOT / directory).glob('*.py'):
            tree = ast.parse(source.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    assert ast.get_docstring(node), f'{source.name}:{node.lineno} needs a docstring'
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert node.returns is not None, (
                        f'{source.name}:{node.lineno} needs a return annotation'
                    )
                    for argument in (
                        *node.args.posonlyargs,
                        *node.args.args,
                        *node.args.kwonlyargs,
                    ):
                        if argument.arg not in ('self', 'cls'):
                            assert argument.annotation is not None, (
                                f'{source.name}:{node.lineno} needs parameter annotations'
                            )
    for name in ('_io', '_model', '_science', '_grades', 'qc', 'eval', 'cli'):
        module = importlib.import_module(f'methridgeqc.{name}')
        for function in vars(module).values():
            if inspect.isfunction(function) and function.__module__ == module.__name__:
                assert typing.get_type_hints(function)
