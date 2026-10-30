"""Check real chr22 file formats, sequence access, and counts independently."""
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from methridgeqc._io import Reference

ROOT = Path(__file__).resolve().parents[1]


def test_chr22_resources() -> None:
    """Check all file formats and sample the first, middle, and last rows."""
    folder = ROOT / 'example'
    for name, columns in [('BS1_D6_1.parquet.lz4', 7), ('D6_reference.parquet.lz4', 136)]:
        path = folder / name
        data = pl.read_parquet(path)
        assert data.width == columns
        assert data.height == {'BS1_D6_1.parquet.lz4': 1144919, 'D6_reference.parquet.lz4': 690864}[name]
        assert data['chrom'].unique().to_list() == ['chr22']
        assert data['end'].eq(data['start'] + 1).all()
        parquet = pq.ParquetFile(path)
        assert all(parquet.metadata.row_group(i).column(j).compression == 'LZ4'
                   for i in range(parquet.num_row_groups) for j in range(columns))
    sequence = b''.join((folder / 'GRCh38_chr22.fa').read_bytes().splitlines()[1:]).upper()
    assert len(sequence) == 50818468
    assert (folder / 'GRCh38_chr22.fa.fai').read_text().split('\t')[:2] == ['chr22', '50818468']
    counts = pl.read_parquet(folder / 'BS1_D6_1.parquet.lz4')
    sample = counts[[0, counts.height // 2, counts.height - 1]]
    assert sample['depth'].eq(sample['m'] + sample['u']).all()
    np.testing.assert_allclose(sample['beta'], sample['m'] / sample['depth'] * 100)
    for start in sample['start']:
        assert 0 <= start < len(sequence)
    indexed = Reference(folder / 'GRCh38_chr22.fa').sequence('chr22')
    for start in (0, len(sequence) // 2, len(sequence) - 100):
        assert indexed[start:start + 100].tobytes() == sequence[start:start + 100]
