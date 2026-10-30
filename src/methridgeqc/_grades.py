"""Canonical grade names and explicit historical-input normalization."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import polars as pl

GRADES: tuple[str, ...] = ('H', 'MH', 'ML', 'L')
ALL_GRADES: tuple[str, ...] = (*GRADES, 'U')
LEGACY_GRADES: dict[str, str] = dict(zip(('A', 'B', 'C', 'D'), GRADES))
GROUP_MEMBERS: dict[str, tuple[str, ...]] = {
    **{g: (g,) for g in ALL_GRADES},
    'H+MH': ('H', 'MH'),
    'ML+L': ('ML', 'L'),
    'MH+ML+L': ('MH', 'ML', 'L'),
}
LEGACY_GROUPS: dict[str, str] = {**LEGACY_GRADES, 'AB': 'H+MH', 'CD': 'ML+L', 'BCD': 'MH+ML+L'}
ORDINAL: dict[str, float] = {g: float(i) for i, g in enumerate(GRADES)} | {'U': np.nan}


def normalize_grades(values: Iterable[object]) -> npt.NDArray[np.str_]:
    """Map historical grades to H/MH/ML/L/U and reject unknown or missing values."""
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or value not in (*ALL_GRADES, *LEGACY_GRADES):
            raise ValueError(f'Invalid QC grade: {value!r}')
        result.append(LEGACY_GRADES.get(value, value))
    return np.asarray(result, dtype='<U2')


def normalize_frame(frame: pl.DataFrame, column: str = 'qc_grade') -> pl.DataFrame:
    """Return a frame with canonical grade labels, preserving all other columns."""
    return frame.with_columns(pl.Series(column, normalize_grades(frame[column].to_list())))


def normalize_groups(frame: pl.DataFrame, column: str = 'group') -> pl.DataFrame:
    """Rename historical group keys without merging potentially conflicting rows."""
    if column not in frame.columns:
        return frame
    values = frame[column].to_list()
    for old, new in LEGACY_GROUPS.items():
        if old in values and new in values:
            raise ValueError(f'Conflicting historical/current group: {old}/{new}')
    return frame.with_columns(pl.Series(column, [LEGACY_GROUPS.get(v, v) for v in values]))


def normalize_keys[ValueT](values: dict[str, ValueT]) -> dict[str, ValueT]:
    """Convert historical statistic keys; reject aliases even when values agree."""
    result: dict[str, ValueT] = {}
    for key, value in values.items():
        head, sep, tail = key.partition('/')
        canonical = LEGACY_GROUPS.get(head, head) + sep + tail
        if canonical in result:
            raise ValueError(f'Conflicting grade keys: {canonical}')
        result[canonical] = value
    return result
