"""Command-line entry points; diagnostics never contaminate stdout."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__

LOGGER: logging.Logger = logging.getLogger(__name__)
LOG_LEVELS: tuple[str, ...] = ('DEBUG', 'INFO', 'WARNING', 'ERROR')


def thread_count(value: str) -> int | None:
    """Parse bounded chromosome workers or the explicit automatic-thread mode."""
    if value == 'auto':
        return None
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            '--threads must be an integer from 1 to 120, or auto'
        ) from exc
    if not 1 <= workers <= 120:
        raise argparse.ArgumentTypeError('--threads must be an integer from 1 to 120, or auto')
    return workers


def parser() -> argparse.ArgumentParser:
    """Build the CLI without importing numerical libraries or changing thread limits."""
    root = argparse.ArgumentParser(
        prog='methridgeqc',
        description='Strand-aware count QC and independent methylation truth evaluation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    root.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    root.add_argument(
        '--log-level', choices=LOG_LEVELS, default='INFO', help='Diagnostic verbosity on stderr'
    )
    commands = root.add_subparsers(dest='command', required=True)
    qc = commands.add_parser('qc', help='Count-based CpG filtering and optional Ridge QC')
    ev = commands.add_parser('eval', help='Evaluate existing frozen QC against percent truth')
    for sub in (qc, ev):
        sub.add_argument(
            '--log-level',
            choices=LOG_LEVELS,
            default=argparse.SUPPRESS,
            help='Diagnostic verbosity on stderr',
        )
        sub.add_argument('--input', required=True, help='Input table, or - for stdin')
        sub.add_argument('--input-format', choices=['tsv', 'parquet', 'bedgraph', 'bedgraph.gz'])
        sub.add_argument('--reference', required=True, help='Uncompressed reference FASTA')
        sub.add_argument('--output', required=True, help='Output file, or - for stdout')
        sub.add_argument('--work-dir', help='Temporary workspace parent; retained on failure')
        sub.add_argument('--chromosomes', nargs='+')
        sub.add_argument('--seed', type=int, default=1729)
    qc.add_argument(
        '--output-format', choices=['bedgraph', 'bedgraph.gz', 'parquet.lz4'], default='bedgraph'
    )
    qc.add_argument('--score', action='store_true')
    qc.add_argument('--classify', action='store_true')
    qc.add_argument('--gpu', action='store_true')
    qc.add_argument(
        '--threads',
        type=thread_count,
        default=1,
        help='Chromosome workers (1..120), or auto without numerical thread limits',
    )
    qc.add_argument('--max-train', type=int, default=200000)
    qc.add_argument('--model-out')
    qc.add_argument('--model-in')
    ev.add_argument('--truth', required=True)
    ev.add_argument('--truth-format', choices=['tsv', 'parquet'])
    ev.add_argument('--truth-scale', choices=['percent', 'fraction'], default='percent')
    ev.add_argument('--bootstrap', type=int, default=300)
    return root


def main() -> int:
    """Run one CLI command with stderr diagnostics and stable success/error exit codes."""
    options = vars(parser().parse_args())
    command = options.pop('command')
    logging.basicConfig(
        level=options.pop('log_level'),
        format='[%(levelname)s] %(name)s: %(message)s',
        stream=sys.stderr,
    )
    # Set bounds before loading numerical libraries in CLI execution.
    if not (command == 'qc' and options['threads'] is None):
        for key in (
            'OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'POLARS_MAX_THREADS',
            'ARROW_NUM_THREADS',
        ):
            os.environ[key] = '1'
    try:
        if command == 'qc':
            from .qc import run
        else:
            from .eval import run
        run(**options)
    except MemoryError:
        LOGGER.error(
            'Insufficient memory for whole-chromosome execution; no row splitting or records omitted'
        )
        LOGGER.debug('Command failure details', exc_info=True)
        return 2
    except (ValueError, RuntimeError, OSError) as exc:
        LOGGER.error('%s', exc)
        LOGGER.debug('Command failure details', exc_info=True)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
