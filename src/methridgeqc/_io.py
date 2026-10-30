"""Streaming interchange and reference validation, with chromosome staging."""

from __future__ import annotations

import gzip
import io
import json
import logging
import shutil
import sqlite3
import sys
import uuid
from collections.abc import Iterator
from numbers import Integral
from pathlib import Path
from typing import Any, Literal, TextIO

import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow.parquet as pq

type InputFormat = Literal['tsv', 'parquet', 'bedgraph', 'bedgraph.gz']
type OutputFormat = Literal['bedgraph', 'bedgraph.gz', 'parquet.lz4']

LOGGER: logging.Logger = logging.getLogger(__name__)
INPUT_FORMATS: tuple[InputFormat, ...] = ('tsv', 'parquet', 'bedgraph', 'bedgraph.gz')
OUTPUT_FORMATS: tuple[OutputFormat, ...] = ('bedgraph', 'bedgraph.gz', 'parquet.lz4')
RAW: list[str] = ['chrom', 'start', 'end', 'm', 'u', 'depth', 'beta']
ANNOTATIONS: list[str] = [
    'strand',
    'canonical_start',
    'qc_valid',
    'qc_reason',
    'qc_risk',
    'qc_risk_quantized',
    'qc_grade',
]
META: dict[str, str] = {
    'format': 'MethRidgeQC_extended_bedgraph_v1',
    'beta_unit': 'percent',
    'risk_direction': 'lower_is_more_reliable',
}


class Reference:
    """Read an uncompressed FASTA through an existing or in-memory FAI index."""

    def __init__(self, path: str | Path) -> None:
        """Validate an existing FAI or scan uniform FASTA wrapping without writing files."""
        self.path = Path(path)
        self.index: dict[str, tuple[int, int, int, int]] = {}
        fai = Path(str(path) + '.fai')
        if fai.exists():
            for line in fai.read_text().splitlines():
                fields = line.split('\t')
                if fields[0] in self.index:
                    raise ValueError('Duplicate FASTA contig')
                self.index[fields[0]] = tuple(map(int, fields[1:5]))
            self.sequences: dict[str, bytes] | None = None
        else:
            # Build an in-memory FAI, not an in-memory genome; never modify FASTA.
            self.sequences = None
            name: str | None = None
            length = offset = bases = width = 0
            previous_bases = 0
            with self.path.open('rb') as stream:
                while line := stream.readline():
                    if line.startswith(b'>'):
                        if name is not None:
                            self.index[name] = (length, offset, bases, width)
                        name = line[1:].split()[0].decode()
                        if name in self.index:
                            raise ValueError('Duplicate FASTA contig')
                        length, offset, bases, width, previous_bases = 0, stream.tell(), 0, 0, 0
                    else:
                        count = len(line.rstrip(b'\r\n'))
                        if name is None or not count or b' ' in line or b'\t' in line:
                            raise ValueError('Invalid FASTA layout')
                        if bases == 0:
                            bases, width = count, len(line)
                        elif previous_bases != bases:
                            raise ValueError('FASTA needs uniform wrapping except final line')
                        length += count
                        previous_bases = count
            if name is not None:
                self.index[name] = (length, offset, bases, width)
        if not self.index:
            raise ValueError('Empty reference')
        LOGGER.debug(
            'Indexed reference %s (%d contigs; existing FAI=%s)',
            self.path,
            len(self.index),
            fai.exists(),
        )

    def sequence(self, chrom: str) -> npt.NDArray[np.uint8]:
        """Load one contig as uppercase nucleotide bytes and verify its indexed length."""
        if chrom not in self.index:
            raise ValueError(f'Unknown reference contig: {chrom}')
        if self.sequences is not None:
            seq = self.sequences[chrom]
        else:
            length, offset, bases, width = self.index[chrom]
            with self.path.open('rb') as stream:
                stream.seek(offset)
                seq = (
                    stream.read(((length - 1) // bases) * width + (length - 1) % bases + 1)
                    .replace(b'\n', b'')
                    .replace(b'\r', b'')
                    .upper()
                )
            if len(seq) != length:
                raise ValueError('FASTA index mismatch')
        return np.frombuffer(seq, dtype=np.uint8)

    def strands(
        self, chrom: str, starts: npt.NDArray[Any], ends: npt.NDArray[Any]
    ) -> npt.NDArray[np.str_]:
        """Validate single-base coordinates and identify reference CpG cytosine strands."""
        seq = self.sequence(chrom)
        try:
            x = starts.astype(float)
            e = ends.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError('Non-numeric coordinate') from exc
        if not np.all(
            np.isfinite(x) & (x == np.floor(x)) & (x >= 0) & (x < len(seq)) & (e == x + 1)
        ):
            raise ValueError(f'Invalid single-base coordinates: {chrom}')
        pos = x.astype(np.int64)
        plus = (
            (pos + 1 < len(seq)) & (seq[pos] == 67) & (seq[np.minimum(pos + 1, len(seq) - 1)] == 71)
        )
        minus = (pos > 0) & (seq[pos] == 71) & (seq[np.maximum(pos - 1, 0)] == 67)
        return np.where(plus, '+', np.where(minus, '-', '?'))


def validate_integer(
    value: object, name: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    """Return a native integer, rejecting booleans, fractions and invalid bounds."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer, not {type(value).__name__}')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be at most {maximum}')
    return int(value)


def input_format(path: str, explicit: InputFormat | None) -> InputFormat:
    """Validate an explicit input format or infer TSV/Parquet from the file suffix."""
    if explicit is not None and explicit not in INPUT_FORMATS:
        raise ValueError(f'Unsupported input format {explicit!r}; choose from {INPUT_FORMATS}')
    if explicit:
        return explicit
    return 'parquet' if path.endswith(('.parquet', '.parquet.lz4')) else 'tsv'


def batches(
    path: str, fmt: InputFormat | None = None, work: Path | None = None
) -> Iterator[pl.DataFrame]:
    """Yield bounded input batches; stage Parquet stdin once in the run workspace."""
    kind = input_format(path, fmt)
    if kind == 'parquet':
        source: Any = path
        if path == '-':
            if work is None:
                raise ValueError('Parquet stdin requires local work directory')
            source = work / 'stdin.parquet'
            with source.open('wb') as output:
                shutil.copyfileobj(sys.stdin.buffer, output)
        for batch in pq.ParquetFile(source).iter_batches(batch_size=65536, use_threads=False):
            yield pl.from_arrow(batch)
        return
    stream: TextIO
    stream = (
        sys.stdin if path == '-' else gzip.open(path, 'rt') if path.endswith('.gz') else open(path)
    )
    try:
        columns: list[str] | None = None
        lines: list[str] = []
        for line in stream:
            if line.startswith('#'):
                if line.startswith('#columns\t'):
                    columns = line.rstrip('\n').split('\t')[1:]
                continue
            if not line.strip():
                continue
            if columns is None:
                columns = line.rstrip('\n').split('\t')
                if len(set(columns)) != len(columns):
                    raise ValueError('Duplicate input column names')
                continue
            lines.append(line)
            if len(lines) >= 65536:
                yield pl.read_csv(
                    io.StringIO(''.join(lines)),
                    separator='\t',
                    has_header=False,
                    new_columns=columns,
                    infer_schema_length=None,
                    null_values=['NA', 'null'],
                )
                lines = []
        if lines:
            yield pl.read_csv(
                io.StringIO(''.join(lines)),
                separator='\t',
                has_header=False,
                new_columns=columns,
                infer_schema_length=None,
                null_values=['NA', 'null'],
            )
        if columns is None:
            raise ValueError('Missing input header')
    finally:
        if path != '-':
            stream.close()


def numeric(frame: pl.DataFrame, name: str) -> npt.NDArray[np.float64]:
    """Convert a numeric column to float64, retaining nulls as NaN."""
    try:
        if frame.schema[name] == pl.Null:
            return np.full(len(frame), np.nan)
        if not frame.schema[name].is_numeric():
            raise ValueError(f'Column {name} must have numeric type')
        return frame[name].cast(pl.Float64).to_numpy()
    except (TypeError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f'Invalid numeric column: {name}') from exc


def legal_counts(frame: pl.DataFrame) -> npt.NDArray[np.bool_]:
    """Identify finite non-negative integer counts satisfying depth = m + u."""
    m, u, n = (numeric(frame, k) for k in ('m', 'u', 'depth'))
    return (
        np.isfinite(m)
        & np.isfinite(u)
        & np.isfinite(n)
        & (m >= 0)
        & (u >= 0)
        & (n >= 0)
        & (m == np.floor(m))
        & (u == np.floor(u))
        & (n == np.floor(n))
        & (m + u == n)
    )


class Workspace:
    """Own a unique run directory that is cleaned only after successful execution."""

    def __init__(self, parent: str | Path | None = None) -> None:
        """Create a unique run directory under the supplied or default workspace parent."""
        base = Path(parent) if parent else Path.cwd() / '.methridgeqc-work'
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / ('run-' + uuid.uuid4().hex)
        self.path.mkdir()

    def finish(self) -> None:
        """Remove this successful run directory and its parent only when empty."""
        shutil.rmtree(self.path)
        try:
            self.path.parent.rmdir()
        except OSError:
            pass


def stage(
    path: str,
    fmt: InputFormat | None,
    reference: Reference,
    work: Path,
    chromosomes: list[str] | None = None,
    qc: bool = True,
) -> tuple[dict[str, Path], dict[str, int]]:
    """Validate reference/count contracts and stage deterministic chromosome files."""
    writers: dict[str, pq.ParquetWriter] = {}
    paths: dict[str, Path] = {}
    seen = sqlite3.connect(work / 'keys.sqlite')
    seen.execute('PRAGMA temp_store=MEMORY')
    seen.execute(
        'CREATE TABLE seen(chrom TEXT,start INTEGER,PRIMARY KEY(chrom,start)) WITHOUT ROWID'
    )
    stats = {
        'input_rows': 0,
        'non_cpg': 0,
        'excluded_contig': 0,
        'invalid_counts': 0,
        'zero_depth': 0,
    }
    if chromosomes is not None and not set(chromosomes) <= set(reference.index):
        seen.close()
        raise ValueError('Requested chromosomes absent from reference')
    try:
        for frame in batches(path, fmt, work):
            missing = set(RAW) - set(frame.columns)
            if missing:
                raise ValueError(f'Missing input columns: {sorted(missing)}')
            if qc and set(ANNOTATIONS) & set(frame.columns):
                raise ValueError('Input annotation field conflict')
            beta = numeric(frame, 'beta')
            if not np.all(np.isfinite(beta) & (beta >= 0) & (beta <= 100)):
                raise ValueError('Input beta must be finite percent in [0,100]')
            stats['input_rows'] += len(frame)
            for key, group in frame.partition_by('chrom', as_dict=True).items():
                chrom = key[0]
                if not isinstance(chrom, str) or chrom not in reference.index:
                    raise ValueError(f'Unknown reference contig: {chrom}')
                if chromosomes is not None and chrom not in chromosomes:
                    stats['excluded_contig'] += len(group)
                    continue
                strands = reference.strands(
                    chrom, group['start'].to_numpy(), group['end'].to_numpy()
                )
                starts = group['start'].cast(pl.Int64).to_numpy()
                try:
                    seen.executemany(
                        'INSERT INTO seen VALUES(?,?)', ((chrom, int(x)) for x in starts)
                    )
                    seen.commit()
                except sqlite3.IntegrityError as exc:
                    raise ValueError(f'Duplicate (chrom,start): {chrom}') from exc
                if (
                    not qc
                    and 'strand' in group.columns
                    and not np.all(group['strand'].to_numpy() == strands)
                ):
                    raise ValueError('QC strand does not match reference')
                stats['non_cpg'] += int(np.sum(strands == '?'))
                keep = strands != '?'
                group = group.filter(pl.Series(keep))
                strands, starts = strands[keep], starts[keep]
                legal = legal_counts(group)
                depth = numeric(group, 'depth')
                valid = legal & (depth > 0)
                stats['invalid_counts'] += int(np.sum(~legal))
                stats['zero_depth'] += int(np.sum(legal & (depth == 0)))
                reason = np.where(~legal, 'invalid_counts', np.where(depth == 0, 'zero_depth', ''))
                if qc:
                    group = group.with_columns(
                        pl.Series('strand', strands),
                        pl.Series('canonical_start', starts - (strands == '-')),
                        pl.Series('qc_valid', valid),
                        pl.Series('qc_reason', reason),
                    )
                elif 'strand' not in group.columns:
                    group = group.with_columns(pl.Series('strand', strands))
                table = group.to_arrow()
                if chrom not in writers:
                    destination = work / f'chrom-{len(paths):06d}.parquet'
                    paths[chrom] = destination
                    writers[chrom] = pq.ParquetWriter(destination, table.schema, compression='lz4')
                # Batches may infer null-only columns differently; freeze the first schema.
                writers[chrom].write_table(table.cast(writers[chrom].schema))
    finally:
        for writer in writers.values():
            writer.close()
        seen.close()
    return dict(sorted(paths.items())), stats


class Output:
    """Write extended bedGraph or LZ4 Parquet without mixing diagnostics into stdout."""

    def __init__(self, path: str, fmt: OutputFormat) -> None:
        """Validate the output format and open text/gzip streams without writing a header."""
        if fmt not in OUTPUT_FORMATS:
            raise ValueError(f'Unsupported output format {fmt!r}; choose from {OUTPUT_FORMATS}')
        self.path, self.fmt = path, fmt
        self.writer: pq.ParquetWriter | None = None
        self.stream: Any = None
        self.header = False
        if fmt != 'parquet.lz4':
            if fmt == 'bedgraph.gz':
                binary = sys.stdout.buffer if path == '-' else open(path, 'wb')
                self.stream = io.TextIOWrapper(
                    gzip.GzipFile(fileobj=binary, mode='wb'), encoding='utf-8'
                )
                self.binary = binary
            else:
                self.stream = sys.stdout if path == '-' else open(path, 'w')

    def write(self, frame: pl.DataFrame) -> None:
        """Append one chromosome frame while preserving measurement values and metadata."""
        if self.fmt == 'parquet.lz4':
            table = frame.to_arrow()
            table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in META.items()})
            if self.writer is None:
                self.writer = pq.ParquetWriter(
                    sys.stdout.buffer if self.path == '-' else self.path,
                    table.schema,
                    compression='lz4',
                )
            self.writer.write_table(table.cast(self.writer.schema))
        else:
            columns = ['chrom', 'start', 'end', 'beta', 'm', 'u', 'depth'] + [
                c for c in frame.columns if c not in RAW
            ]
            if not self.header:
                self.stream.write(
                    '#' + json.dumps(META) + '\n#columns\t' + '\t'.join(columns) + '\n'
                )
                self.header = True
            # Polars may write directly to fileno, bypassing gzip's compression wrapper.
            self.stream.write(
                frame.select(columns).write_csv(
                    separator='\t', include_header=False, null_value='NA'
                )
            )

    def close(self, empty: pl.DataFrame | None = None) -> None:
        """Finalize the output, creating an empty schema when requested."""
        if not self.header and self.writer is None and empty is not None:
            self.write(empty)
        if self.writer is not None:
            self.writer.close()
        if self.stream is not None:
            self.stream.flush()
            if self.fmt == 'bedgraph.gz':
                self.stream.detach().close()
                if self.path != '-':
                    self.binary.close()
            elif self.path != '-':
                self.stream.close()
