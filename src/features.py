"""
Feature Engineering
===================
Transforms the raw training_data.csv into a model-ready feature matrix.

Features produced
-----------------
Numerical:
  hour                        — 0-23
  day_of_week                 — 0 (Mon) … 6 (Sun)
  is_weekend                  — 1 if day_of_week ∈ {5, 6}
  is_rush_hour                — 1 if hour ∈ morning (6-9) or evening (16-19) rush
  is_night                    — 1 if hour ∈ 0-5 or 22-23
  hist_avg_delay_station      — mean delay at this station across the training set
  hist_delay_rate_station     — fraction of departures > 5 min at this station
  hist_avg_delay_train_type   — mean delay for this train type

Categorical (encoded):
  train_type_encoded          — OrdinalEncoder (order by avg delay, worst = highest)
  station_encoded             — OrdinalEncoder (order by avg delay)

Target:
  is_delayed                  — bool → int (0 / 1)

Public API
----------
  build_features(df)          → (X: DataFrame, y: Series, encoders: dict)
  apply_features(df, encoders)→  X: DataFrame   (for inference, no target needed)
  save_encoders(encoders, path)
  load_encoders(path)         → encoders: dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rush-hour / time-window definitions
# ---------------------------------------------------------------------------
MORNING_RUSH = set(range(6, 10))    # 06:00 – 09:59
EVENING_RUSH = set(range(16, 20))   # 16:00 – 19:59
NIGHT_HOURS  = set(range(0, 6)) | {22, 23}

# Train-type ordering by known punctuality (worst → best → numeric 0 = worst)
# This gives the model an ordinal signal rather than arbitrary label noise.
TRAIN_TYPE_ORDER = ["IC", "EC", "ICE", "TGV", "EN", "NJ", "IRE", "RE", "RB", "S"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic time-window boolean features."""
    df = df.copy()
    df["is_weekend"]   = df["day_of_week"].isin({5, 6}).astype(int)
    df["is_rush_hour"] = df["hour"].isin(MORNING_RUSH | EVENING_RUSH).astype(int)
    df["is_night"]     = df["hour"].isin(NIGHT_HOURS).astype(int)
    return df


def _historical_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """
    Compute leave-group aggregates for station and train_type.

    Returns the augmented DataFrame and a lookup dict that can be reused
    at inference time.
    """
    station_stats = (
        df.groupby("station_name")
        .agg(
            hist_avg_delay_station=("delay_minutes", "mean"),
            hist_delay_rate_station=("is_delayed", "mean"),
        )
    )
    train_type_stats = (
        df.groupby("train_type")
        .agg(hist_avg_delay_train_type=("delay_minutes", "mean"))
    )

    df = df.join(station_stats, on="station_name")
    df = df.join(train_type_stats, on="train_type")

    # Global fallbacks for unseen values at inference time
    global_avg   = df["delay_minutes"].mean()
    global_rate  = df["is_delayed"].mean()

    lookups = {
        "station_avg_delay":    station_stats["hist_avg_delay_station"],
        "station_delay_rate":   station_stats["hist_delay_rate_station"],
        "train_type_avg_delay": train_type_stats["hist_avg_delay_train_type"],
        "global_avg_delay":     global_avg,
        "global_delay_rate":    global_rate,
    }
    return df, lookups


def _encode_train_type(
    series: pd.Series,
    order: list[str] | None = None,
    fitted_map: dict[str, int] | None = None,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Map train_type to an integer 0-N (0 = historically worst punctuality).
    Unknown values at inference fall back to the median code.
    """
    if fitted_map is not None:
        fallback = int(np.median(list(fitted_map.values())))
        return series.map(fitted_map).fillna(fallback).astype(int), fitted_map

    order = order or TRAIN_TYPE_ORDER
    # Only include types present in this dataset
    present = [t for t in order if t in series.unique()]
    # Anything not in the canonical order gets appended
    extras = [t for t in series.unique() if t not in order]
    full_order = present + extras
    mapping = {t: i for i, t in enumerate(full_order)}
    return series.map(mapping).fillna(0).astype(int), mapping


def _encode_station(
    series: pd.Series,
    avg_delay_lookup: pd.Series | None = None,
    fitted_map: dict[str, int] | None = None,
) -> tuple[pd.Series, dict[str, int]]:
    """
    Rank stations by their average delay (worst = highest rank number).
    Unknown stations at inference fall back to the median rank.
    """
    if fitted_map is not None:
        fallback = int(np.median(list(fitted_map.values())))
        return series.map(fitted_map).fillna(fallback).astype(int), fitted_map

    assert avg_delay_lookup is not None
    ranked = avg_delay_lookup.rank(method="first").astype(int)
    mapping: dict[str, int] = {str(k): int(v) for k, v in ranked.to_dict().items()}
    fallback = int(np.median(list(mapping.values())))
    return pd.Series(series.map(mapping).fillna(fallback), dtype=int), mapping


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
    "is_night",
    "train_type_encoded",
    "station_encoded",
    "hist_avg_delay_station",
    "hist_delay_rate_station",
    "hist_avg_delay_train_type",
]


def build_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """
    Full feature-engineering pipeline for training.

    Parameters
    ----------
    df : raw DataFrame from collect_data.py (must have all canonical columns)

    Returns
    -------
    X        : feature DataFrame  (shape: n_rows × len(FEATURE_COLUMNS))
    y        : target Series (int 0/1)
    encoders : dict of fitted artefacts needed to reproduce features at inference
    """
    df = df.copy()
    df["is_delayed"] = df["is_delayed"].astype(int)

    # 1. Time-window flags
    df = _time_features(df)

    # 2. Historical aggregates (computed on the whole training set)
    df, lookups = _historical_stats(df)

    # 3. Categorical encodings
    df["train_type_encoded"], tt_map = _encode_train_type(df["train_type"])
    df["station_encoded"],    st_map = _encode_station(
        df["station_name"], avg_delay_lookup=lookups["station_avg_delay"]
    )

    encoders = {
        "train_type_map":       tt_map,
        "station_map":          st_map,
        "station_avg_delay":    lookups["station_avg_delay"],
        "station_delay_rate":   lookups["station_delay_rate"],
        "train_type_avg_delay": lookups["train_type_avg_delay"],
        "global_avg_delay":     lookups["global_avg_delay"],
        "global_delay_rate":    lookups["global_delay_rate"],
        "feature_columns":      FEATURE_COLUMNS,
    }

    X = df[FEATURE_COLUMNS]
    y = df["is_delayed"]
    return X, y, encoders


def apply_features(
    df: pd.DataFrame,
    encoders: dict[str, Any],
) -> pd.DataFrame:
    """
    Apply fitted encoders to new data (inference / API use).

    Parameters
    ----------
    df       : DataFrame with at minimum: station_name, train_type, hour, day_of_week
    encoders : dict returned by build_features() or loaded with load_encoders()

    Returns
    -------
    X : feature DataFrame ready for model.predict()
    """
    df = df.copy()

    # Time flags
    df = _time_features(df)

    # Historical lookups — use fitted stats, fall back to global averages for unknowns
    df["hist_avg_delay_station"] = (
        df["station_name"]
        .map(encoders["station_avg_delay"])
        .fillna(encoders["global_avg_delay"])
    )
    df["hist_delay_rate_station"] = (
        df["station_name"]
        .map(encoders["station_delay_rate"])
        .fillna(encoders["global_delay_rate"])
    )
    df["hist_avg_delay_train_type"] = (
        df["train_type"]
        .map(encoders["train_type_avg_delay"])
        .fillna(encoders["global_avg_delay"])
    )

    # Categorical encodings
    df["train_type_encoded"], _ = _encode_train_type(
        df["train_type"], fitted_map=encoders["train_type_map"]
    )
    df["station_encoded"], _ = _encode_station(
        df["station_name"], fitted_map=encoders["station_map"]
    )

    return df[encoders["feature_columns"]]


def save_encoders(encoders: dict[str, Any], path: str | Path) -> None:
    joblib.dump(encoders, path)
    print(f"Encoders saved → {path}")


def load_encoders(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "data" / "training_data.csv")
    print(f"Input: {df.shape}")

    X, y, encoders = build_features(df)
    print(f"X shape: {X.shape}   y shape: {y.shape}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Class balance: {y.value_counts().to_dict()}")
    print(f"\nFirst row:\n{X.iloc[0]}")

    # Round-trip test: apply_features on a slice
    sample = df[["station_name", "train_type", "hour", "day_of_week"]].head(5)
    X2 = apply_features(sample, encoders)
    print(f"\napply_features smoke-test:\n{X2}")
