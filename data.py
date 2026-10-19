"""Hourly data, chronological windows and geographic topology."""
from __future__ import annotations

# ======================================================================
"""Geographical distances and adjacency for an explicit sensor topology."""

import numpy as np

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two GPS points."""
    R = 6_371_000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def compute_distance_matrix(node_coords: np.ndarray) -> np.ndarray:
    """Returns pairwise Haversine distance matrix (N x N) in meters."""
    node_coords = np.asarray(node_coords, dtype=np.float64)
    n = len(node_coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_distance(
                node_coords[i, 0], node_coords[i, 1],
                node_coords[j, 0], node_coords[j, 1]
            )
            D[i, j] = d
            D[j, i] = d
    return D


def compute_physical_adjacency(
    sigma: float = None,
    node_indices=None,
    node_coords: np.ndarray = None,
) -> np.ndarray:
    """
    Computes a Gaussian-kernel adjacency matrix from physical GPS coordinates.

    A_ij = exp(-dist(i,j)^2 / sigma^2)

    If sigma is None, it defaults to the standard deviation of all pairwise distances.
    """
    if node_coords is None:
        raise ValueError("Sensor coordinates are required for a physical adjacency")
    D = compute_distance_matrix(node_coords)
    if node_indices is not None:
        D = D[np.ix_(node_indices, node_indices)]
    if sigma is None:
        # Use std of upper-triangle distances as bandwidth
        upper_dists = D[np.triu_indices_from(D, k=1)]
        if len(upper_dists):
            spread = float(np.std(upper_dists))
            # Two nodes have zero distance spread. Their actual separation is
            # the meaningful scale; 1 m would underflow a distant edge.
            sigma = max(spread if spread >= 1.0 else float(np.median(upper_dists)), 1.0)
        else:
            sigma = 1.0

    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Physical graph bandwidth must be positive and finite")

    A = np.exp(-(D / sigma) ** 2)
    np.fill_diagonal(A, 1.0)
    return A.astype(np.float32)


# ======================================================================
"""
Data loading and preprocessing pipeline for LoRaWAN RSSI forecasting.

Ensures strict zero-leakage:
- Chronological train/val/test split.
- Windows containing missing values are discarded (no imputation).
- Windows must be strictly hourly and cannot cross acquisition gaps.
- Scalers fitted SOLELY on the training set.
"""

from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DEFAULT_DATA_FILE = "data/uva_gatewayA_hourly.csv"
TARGET_COLS = [f"RSSI_{i:02d}" for i in range(1, 9)]
EXOGENOUS_COLS = ["temp", "hum", "bar", "rain"]


def load_raw_data(
    file_path: str = DEFAULT_DATA_FILE,
) -> pd.DataFrame:
    """Load the explicitly prepared local hourly dataset."""
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {file_path}; run lora-prepare-uva")
    df = pd.read_csv(file_path, sep=";")

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def split_chronological(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series chronologically into train, validation, and test sets.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


class LoRaDataPipeline:
    """Handles training-only scaling while preserving missing observations."""
    def __init__(
        self,
        target_cols: List[str] = TARGET_COLS,
        exogenous_cols: List[str] = EXOGENOUS_COLS
    ):
        self.target_cols = target_cols
        self.exogenous_cols = exogenous_cols
        self.all_cols = target_cols + exogenous_cols
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame):
        """Fit one scaler per feature using observed training values only."""
        for col in self.all_cols:
            observed = train_df[[col]].dropna()
            if observed.empty:
                raise ValueError(f"Training column {col!r} has no observed values.")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(observed.to_numpy(dtype=np.float64))
            self.scalers[col] = scaler

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale observed values and preserve missing values for window filtering."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted on train data before transform.")

        df_scaled = pd.DataFrame(index=df.index)
        for col in self.all_cols:
            values = df[[col]].to_numpy(dtype=np.float64)
            valid = np.isfinite(values[:, 0])
            scaled = np.full(len(values), np.nan, dtype=np.float64)
            if valid.any():
                scaled[valid] = self.scalers[col].transform(values[valid]).ravel()
            df_scaled[col] = scaled
        return df_scaled

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        """
        Inverse transform target predictions or ground truth back to physical dBm.
        scaled_targets shape: (batch, pred_len, n_targets) or (batch, n_targets)
        """
        orig_shape = scaled_targets.shape
        if len(orig_shape) == 2:
            scaled_targets = scaled_targets[:, np.newaxis, :]

        batch_size, pred_len, n_targets = scaled_targets.shape
        unscaled = np.zeros_like(scaled_targets, dtype=np.float32)

        for i, col in enumerate(self.target_cols):
            col_data = scaled_targets[:, :, i].reshape(-1, 1)
            unscaled[:, :, i] = self.scalers[col].inverse_transform(col_data).reshape(batch_size, pred_len)

        if len(orig_shape) == 2:
            return unscaled.squeeze(axis=1)
        return unscaled


def create_sliding_windows(
    data: np.ndarray,
    seq_length: int,
    pred_length: int,
    n_targets: int,
    timestamps: Optional[np.ndarray] = None,
    expected_frequency: np.timedelta64 = np.timedelta64(1, "h"),
    return_origins: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates strictly causal sliding windows:
    X: [t - seq_length, ..., t - 1] (all features)
    y: [t, ..., t + pred_length - 1] (target RSSIs only)
    """
    X, y, origins = [], [], []
    total_steps = len(data) - seq_length - pred_length + 1
    for i in range(max(0, total_steps)):
        window = data[i : i + seq_length + pred_length]
        if not (np.isfinite(window[:seq_length]).all()
                and np.isfinite(window[seq_length:, :n_targets]).all()):
            continue
        if timestamps is not None:
            time_window = timestamps[i : i + seq_length + pred_length]
            if not np.all(np.diff(time_window) == expected_frequency):
                continue
        origins.append(i + seq_length)
        X.append(window[:seq_length])
        y.append(window[seq_length:, :n_targets])

    if not X:
        result = (
            np.empty((0, seq_length, data.shape[1]), dtype=np.float32),
            np.empty((0, pred_length, n_targets), dtype=np.float32),
        )
    else:
        result = (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32))
    return (*result, np.asarray(origins, dtype=int)) if return_origins else result


def _timestamps(df: pd.DataFrame) -> np.ndarray:
    """Parse Unix-second or textual timestamps to nanosecond datetime values."""
    values = df["timestamp"]
    if pd.api.types.is_numeric_dtype(values):
        parsed = pd.to_datetime(values, unit="s", utc=True)
    else:
        parsed = pd.to_datetime(values, utc=True)
    if parsed.isna().any():
        raise ValueError("Dataset contains invalid timestamps.")
    return parsed.to_numpy(dtype="datetime64[ns]")


def get_prepared_datasets(
    file_path: str = DEFAULT_DATA_FILE,
    seq_length: int = 24,
    pred_length: int = 1,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
):
    """
    Build chronological datasets from complete, strictly hourly windows.
    """
    raw_df = load_raw_data(file_path=file_path)
    target_cols = sorted(
        (column for column in raw_df.columns if column.startswith("RSSI_")),
        key=lambda column: int(column.split("_")[-1]),
    )
    if not target_cols:
        raise ValueError("Dataset must contain columns named RSSI_01, RSSI_02, ...")
    missing_exogenous = set(EXOGENOUS_COLS) - set(raw_df.columns)
    if missing_exogenous:
        raise ValueError(f"Dataset is missing exogenous columns {sorted(missing_exogenous)}")
    train_df, val_df, test_df = split_chronological(raw_df, train_ratio, val_ratio)

    pipeline = LoRaDataPipeline(target_cols, EXOGENOUS_COLS)
    pipeline.fit(train_df)

    train_scaled = pipeline.transform(train_df)
    val_scaled = pipeline.transform(val_df)
    test_scaled = pipeline.transform(test_df)

    n_targets = len(target_cols)
    X_train, y_train, train_idx = create_sliding_windows(
        train_scaled.values, seq_length, pred_length, n_targets, _timestamps(train_df), return_origins=True
    )
    X_val, y_val, val_idx = create_sliding_windows(
        val_scaled.values, seq_length, pred_length, n_targets, _timestamps(val_df), return_origins=True
    )
    X_test, y_test, test_idx = create_sliding_windows(
        test_scaled.values, seq_length, pred_length, n_targets, _timestamps(test_df), return_origins=True
    )

    for split_name, X in (("train", X_train), ("validation", X_val), ("test", X_test)):
        if len(X) == 0:
            raise ValueError(
                f"No valid {split_name} windows remain after removing missing/non-hourly intervals."
            )

    # Statistical estimators reset conditional innovation states at every gap.
    train_times = _timestamps(train_df)
    train_segments = []
    current = []
    previous_time = None
    for row, timestamp in zip(train_scaled.to_numpy(), train_times):
        valid = np.isfinite(row).all()
        consecutive = previous_time is None or timestamp-previous_time == np.timedelta64(1, "h")
        if not valid or not consecutive:
            if current:
                train_segments.append(np.asarray(current))
            current = []
        if valid:
            current.append(row)
        previous_time = timestamp if valid else None
    if current:
        train_segments.append(np.asarray(current))

    return {
        "training_segments": train_segments,
        "pipeline": pipeline,
        "target_scale": np.array([1.0 / pipeline.scalers[c].scale_[0] for c in target_cols]),
        "origin_timestamps": {"train": _timestamps(train_df)[train_idx],
                              "val": _timestamps(val_df)[val_idx],
                              "test": _timestamps(test_df)[test_idx]},
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "target_names": target_cols,
        "feature_names": pipeline.all_cols,
        "topology": _load_topology(file_path, target_cols),
        "quality_report": {
            "policy": "complete strictly-hourly windows; no imputation",
            "valid_windows": {
                "train": len(X_train), "validation": len(X_val), "test": len(X_test)
            },
            "discarded_windows": {
                "train": max(0, len(train_df) - seq_length - pred_length + 1) - len(X_train),
                "validation": max(0, len(val_df) - seq_length - pred_length + 1) - len(X_val),
                "test": max(0, len(test_df) - seq_length - pred_length + 1) - len(X_test),
            },
        }
    }



ROOT = Path(__file__).resolve().parent
EXPECTED_WINDOWS = {'A': {'train': 7145, 'validation': 1369, 'test': 1015},
                    'C': {'train': 4967, 'validation': 1052, 'test': 1161}}


def _load_topology(file_path, target_cols):
    sidecar = Path(file_path).with_name('topology.csv')
    frame = pd.read_csv(sidecar, float_precision='round_trip')
    if frame['node'].tolist() != target_cols:
        raise ValueError('Topology target order differs from the hourly data')
    coords = frame[['latitude', 'longitude']].to_numpy(dtype=np.float64)
    receivers = frame[['gateway_latitude', 'gateway_longitude']].to_numpy(dtype=np.float64)
    if not np.isfinite(coords).all() or not np.isfinite(receivers).all():
        raise ValueError('Nonfinite coordinates')
    if not np.all(receivers == receivers[0]):
        raise ValueError('Each task must have one receiver')
    receiver = receivers[0]
    distances = np.asarray([haversine_distance(*point, *receiver) for point in coords])
    return dict(node_coordinates=coords, gateway_coordinates=receiver,
                gateway_distances_m=distances, source=str(sidecar))


def load_configs(gateway):
    import ast
    import csv
    configs = {}
    with (ROOT / 'configs' / f'{gateway}.csv').open() as stream:
        for row in csv.DictReader(stream):
            config = configs.setdefault(row['model'], {})
            if row['parameter'] in config:
                raise ValueError('Duplicate configuration entry')
            config[row['parameter']] = ast.literal_eval(row['value'])
    return configs


def prepare(gateway):
    data = get_prepared_datasets(str(ROOT / 'data' / gateway / 'hourly.csv'), 24, 1)
    if data['quality_report']['valid_windows'] != EXPECTED_WINDOWS[gateway]:
        raise ValueError('Window counts differ from the paper protocol')
    return data
