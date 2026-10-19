"""Per-link errors, gateway summaries and paired temporal uncertainty."""
from __future__ import annotations

# ======================================================================
"""Paired, gap-aware descriptive intervals from saved H=1 predictions.

Negative candidate-minus-reference MAE differences favor the candidate.
This script never trains a model or selects one from test results.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def contiguous_runs(timestamps: np.ndarray) -> list[np.ndarray]:
    """Partition forecast targets where successive timestamps differ by one hour."""
    breaks = np.flatnonzero(np.diff(timestamps) != np.timedelta64(1, "h")) + 1
    return [part for part in np.split(np.arange(len(timestamps)), breaks)
            if len(part)]


def block_intervals(loss_difference: np.ndarray, timestamps: np.ndarray,
                    block_hours: int = 24, repeats: int = 2000,
                    seed: int = 2026) -> tuple[np.ndarray, np.ndarray]:
    """Circular block resampling within each observed contiguous run."""
    if block_hours < 1 or repeats < 1 or len(loss_difference) != len(timestamps):
        raise ValueError("Invalid block bootstrap input")
    runs = contiguous_runs(timestamps)
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, loss_difference.shape[1]), dtype=np.float64)
    for repetition in range(repeats):
        sampled = []
        for run in runs:
            length = len(run)
            width = min(block_hours, length)
            pieces = []
            while sum(len(piece) for piece in pieces) < length:
                start = rng.integers(length)
                pieces.append(run[(start + np.arange(width)) % length])
            sampled.append(np.concatenate(pieces)[:length])
        boot[repetition] = loss_difference[np.concatenate(sampled)].mean(axis=0)
    return np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)


import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

METADATA = ['target_time', 'forecast_origin', 'node', 'observed']
CONTRASTS = [('SingleNode_Seq2Seq', 'MultiNode_Seq2Seq'),
             ('STGNN_NoGraph', 'PhysicalAdaptive_STGNN'),
             ('STGNN_AdaptiveOnly', 'PhysicalAdaptive_STGNN'),
             ('STGNN_PhysicalOnly', 'PhysicalAdaptive_STGNN')]


def arrays_from_csv(path):
    """Restore full-precision, common-origin arrays from a readable CSV archive."""
    frame = pd.read_csv(path, float_precision='round_trip')
    if not set(METADATA).issubset(frame.columns):
        raise ValueError('Missing prediction metadata')
    frame['target_time'] = pd.to_datetime(frame['target_time'])
    frame['forecast_origin'] = pd.to_datetime(frame['forecast_origin'])
    frame = frame.sort_values(['target_time', 'node'])
    times = frame['target_time'].drop_duplicates().to_numpy(dtype='datetime64[ns]')
    names = sorted(frame['node'].unique())
    if frame.duplicated(['target_time', 'node']).any() or len(frame) != len(times) * len(names):
        raise ValueError('Incomplete or duplicate common forecast origins')
    if not (frame['target_time'] - frame['forecast_origin'] == pd.Timedelta(hours=1)).all():
        raise ValueError('Forecast origins must precede targets by one hour')
    expected = pd.MultiIndex.from_product([times, names], names=['target_time', 'node'])
    frame = frame.set_index(['target_time', 'node']).reindex(expected)
    models = [c for c in frame.columns if c not in METADATA]
    if not np.isfinite(frame[['observed'] + models].to_numpy()).all():
        raise ValueError('Nonfinite observations or predictions')
    shape = (len(times), 1, len(names))
    return frame['observed'].to_numpy(dtype=np.float32).reshape(shape), {
        name: frame[name].to_numpy(dtype=np.float32).reshape(shape) for name in models}, names, times


def summarize(truth, predictions, names, times, output, *, paired=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, macros = [], []
    for model, prediction in predictions.items():
        if prediction.shape != truth.shape or not np.isfinite(prediction).all():
            raise ValueError(f'Invalid predictions for {model}')
        error = prediction - truth
        mae = np.abs(error).mean(axis=(0, 1))
        rmse = np.sqrt(np.square(error).mean(axis=(0, 1)))
        rows.extend(dict(model=model, node=node,
                         mae_db=float(mean_absolute_error(truth[:, :, i].flatten(), prediction[:, :, i].flatten())),
                         rmse_db=float(root_mean_squared_error(truth[:, :, i].flatten(), prediction[:, :, i].flatten())))
                    for i, node in enumerate(names))
        macros.append(dict(model=model, mae_db=float(mae.mean()), rmse_db=float(rmse.mean())))
    pd.DataFrame(rows).to_csv(output / 'per_link.csv', index=False, float_format='%.17g')
    pd.DataFrame(macros).to_csv(output / 'macro.csv', index=False, float_format='%.17g')
    if paired:
        rows = []
        for reference, candidate in CONTRASTS:
            loss = (np.abs(predictions[candidate] - truth) -
                    np.abs(predictions[reference] - truth))[:, 0, :]
            lower, upper = block_intervals(loss, times, 24, 2000, 2026)
            estimates = loss.mean(axis=0)
            rows.extend(dict(reference=reference, candidate=candidate, node=node,
                             delta_mae_db=float(estimates[i]),
                             ci95_low_db=float(lower[i]), ci95_high_db=float(upper[i]))
                        for i, node in enumerate(names))
        pd.DataFrame(rows).to_csv(output / 'paired.csv', index=False, float_format='%.17g')


def verify_reference(output, reference):
    """Compare full-precision results with archived, rounded per-link metrics."""
    for filename, keys, tolerance in [
        ('per_link.csv', ['model', 'node'], 5.01e-5),
        ('macro.csv', ['model'], 1e-12),
        ('paired.csv', ['reference', 'candidate', 'node'], 1e-12),
    ]:
        actual = pd.read_csv(Path(output) / filename, float_precision='round_trip').set_index(keys).sort_index()
        expected = pd.read_csv(Path(reference) / filename, float_precision='round_trip').set_index(keys).sort_index()
        if not actual.index.equals(expected.index) or not actual.columns.equals(expected.columns):
            raise ValueError(f'Unexpected result labels: {filename}')
        np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=0, atol=tolerance)
