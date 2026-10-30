"""Canonical grade migration must preserve numerical results."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from methridgeqc import _science, qc
from methridgeqc._grades import (
    ALL_GRADES,
    GROUP_MEMBERS,
    normalize_grades,
    normalize_groups,
    normalize_keys,
)
from methridgeqc.eval import evaluate


def test_quantizer_boundaries_unchanged() -> None:
    frozen = {
        'values': [0.0, 0.25, 0.5, 1.0],
        'edges': [0.125, 0.375, 0.75],
        'cuts': [0.25, 0.5, 1.0],
        'supported': True,
    }
    risks = np.array([0.0, 0.124, 0.125, 0.374, 0.375, 0.749, 0.75, 1.0])
    quantized, grades = _science.apply_quantizer(risks, frozen)
    np.testing.assert_array_equal(quantized, [0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 1.0, 1.0])
    assert grades.tolist() == ['H', 'H', 'MH', 'MH', 'ML', 'ML', 'L', 'L']
    assert frozen['cuts'] == [0.25, 0.5, 1.0]


def test_qc_multichar_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'input.parquet'
    pl.DataFrame({'start': [0, 1, 2, 3, 4]}).write_parquet(path)
    monkeypatch.setattr(
        qc,
        'chromosome_features',
        lambda frame, anchors: {
            'valid': np.array([True] * 4 + [False]),
            **{k: np.ones(5) for k in qc.FEATURES},
        },
    )
    monkeypatch.setattr(qc._model, 'predict', lambda x, model, gpu: np.array([0.0, 0.25, 0.5, 1.0]))
    model = {
        'quantizer': {
            'values': [0.0, 0.25, 0.5, 1.0],
            'edges': [0.125, 0.375, 0.75],
            'cuts': [0.25, 0.5, 1.0],
            'supported': True,
        }
    }
    output = qc.score_chromosome(('chr1', path), model, True, True, False)
    target = tmp_path / 'qc.parquet'
    output.write_parquet(target)
    assert pl.read_parquet(target)['qc_grade'].to_list() == list(ALL_GRADES)


def test_old_new_eval_identical() -> None:
    frame = pl.DataFrame(
        {
            'chrom': ['chr1'] * 5,
            '_block': [0] * 5,
            'truth_status': ['valid'] * 5,
            'error': [0.01, 0.2, 0.3, 0.5, 0.4],
            'qc_grade': list('ABCDU'),
        }
    )
    old = evaluate(frame, {'chr1': 1}, bootstrap=0)
    new = evaluate(
        frame.with_columns(pl.Series('qc_grade', list(ALL_GRADES))), {'chr1': 1}, bootstrap=0
    )
    assert old.equals(new)
    assert 'H+MH' in old['group'].to_list()
    assert not set('ABCD') & set(old['group'].to_list())
    for invalid in ('M', '', None, 'unknown'):
        with pytest.raises(ValueError):
            evaluate(
                frame.with_columns(pl.Series('qc_grade', ['H', 'MH', 'ML', 'L', invalid])),
                {'chr1': 1},
                bootstrap=0,
            )


def test_members_and_conflicts() -> None:
    assert GROUP_MEMBERS['H+MH'] == ('H', 'MH')
    assert GROUP_MEMBERS['ML+L'] == ('ML', 'L')
    assert GROUP_MEMBERS['MH+ML+L'] == ('MH', 'ML', 'L')
    with pytest.raises(ValueError):
        normalize_groups(pl.DataFrame({'group': ['A', 'H']}))
    with pytest.raises(ValueError):
        normalize_keys({'B': 1, 'MH': 1})
    for value in (None, '', 'M'):
        with pytest.raises(ValueError):
            normalize_grades([value])
