"""Share a real-data subset so logging tests need not process a full chromosome."""
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def chr22_subset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Select the first 256 rows in their original order."""
    target = tmp_path_factory.mktemp('chr22') / 'counts.parquet'
    pl.scan_parquet(ROOT / 'example/BS1_D6_1.parquet.lz4').head(256).collect().write_parquet(target)
    return target


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest temporary files inside the project by default."""
    if config.option.basetemp is None:
        config.option.basetemp = str(ROOT / 'temp/pytest')
    Path(config.option.basetemp).parent.mkdir(parents=True, exist_ok=True)
