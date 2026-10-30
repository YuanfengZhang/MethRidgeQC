"""Keep source metadata and CLI entry points aligned."""

import subprocess
import sys
import tomllib
from pathlib import Path

from methridgeqc import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_version_and_help() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'methridgeqc.cli', '--version'],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == f'methridgeqc {__version__}'
    for arguments in (['--help'], ['qc', '--help'], ['eval', '--help']):
        subprocess.run(
            [sys.executable, '-m', 'methridgeqc.cli', *arguments], check=True, capture_output=True
        )


def test_metadata() -> None:
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert metadata['project']['readme'] == 'README.md'
    assert metadata['project']['requires-python'] == '>=3.13,<3.14'
    assert metadata['tool']['setuptools']['dynamic']['version']['attr'] == 'methridgeqc.__version__'
    assert metadata['project']['scripts']['methridgeqc'] == 'methridgeqc.cli:main'


def test_example_distribution_resources() -> None:
    """Require real example files and the FASTA index in the source distribution."""
    for name in ('BS1_D6_1.parquet.lz4', 'D6_reference.parquet.lz4',
                 'GRCh38_chr22.fa', 'GRCh38_chr22.fa.fai', 'README.md'):
        assert (ROOT / 'example' / name).is_file()
    assert 'recursive-include example *.fa *.fai *.parquet.lz4 *.md' in (ROOT / 'MANIFEST.in').read_text()


def test_gpu_dependency_selection() -> None:
    """Keep default dependencies CPU-only and use CUDA 13 for the GPU extra and Pixi."""
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    default = metadata['project']['dependencies']
    assert not any('cuda' in item or 'cuml' in item or 'cupy' in item for item in default)
    gpu = metadata['project']['optional-dependencies']['gpu']
    assert any(item.startswith('cuml-cu13==26.10.') for item in gpu)
    assert any(item.startswith('rmm-cu13==26.10.') for item in gpu)
    assert any(item.startswith('cupy-cuda13x[ctk]') for item in gpu)
    pixi = tomllib.loads((ROOT / 'pixi.toml').read_text())
    assert pixi['feature']['gpu']['system-requirements']['cuda'] == '13'
    assert pixi['feature']['gpu']['dependencies']['cuda-version'] == '13.*'
    readme = (ROOT / 'README.md').read_text()
    assert "uv pip install '.[gpu]'" in readme
    assert 'proxychains' not in readme
