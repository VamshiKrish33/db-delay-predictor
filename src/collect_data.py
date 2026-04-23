"""
Step 1 — Data Collection
========================
Pulls training data from two sources and merges them into one CSV:

  1. HuggingFace dataset  piebro/deutsche-bahn-data  (historical bulk data)
  2. Project-1 PostgreSQL  db_delays.departures       (recent pipeline data)

Output: data/training_data.csv

Canonical schema after normalisation
--------------------------------------
station_name       str      — station name (title-cased, NaN dropped)
train_type         str      — ICE / IC / RE / RB / S / etc.
planned_departure  datetime — scheduled departure (UTC-aware stripped to naive)
delay_minutes      float    — signed delay in minutes (negative = early)
is_cancelled       bool
hour               int      — 0-23 extracted from planned_departure
day_of_week        int      — 0=Mon … 6=Sun
is_delayed         bool     — True when delay_minutes > 5  (ML target)
source             str      — "huggingface" | "postgres"

Usage
-----
    python src/collect_data.py                   # default: 200 000 HF rows
    python src/collect_data.py --hf-rows 50000   # lighter run for testing
    python src/collect_data.py --no-postgres      # skip DB if not running
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "training_data.csv"

load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Train types we actually care about — filters out trams, buses, etc.
VALID_TRAIN_TYPES = {"ICE", "IC", "EC", "RE", "RB", "S", "IRE", "TGV", "NJ", "EN"}

_CANONICAL_COLUMNS = [
    "station_name",
    "train_type",
    "planned_departure",
    "delay_minutes",
    "is_cancelled",
    "hour",
    "day_of_week",
    "is_delayed",
    "source",
]


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day_of_week, and is_delayed from planned_departure / delay_minutes."""
    df["planned_departure"] = pd.to_datetime(df["planned_departure"], utc=False, errors="coerce")
    # Strip timezone so both sources are comparable
    df["planned_departure"] = df["planned_departure"].dt.tz_localize(None)
    df["hour"] = df["planned_departure"].dt.hour
    df["day_of_week"] = df["planned_departure"].dt.dayofweek
    df["is_delayed"] = df["delay_minutes"] > 5
    return df


def _clean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalise, filter, and reorder to canonical schema."""
    df["source"] = source

    # Drop rows with no station or no departure time
    df = df.dropna(subset=["station_name", "planned_departure"])

    # Keep only DB rail train types
    df = df[df["train_type"].isin(VALID_TRAIN_TYPES)]

    df["station_name"] = df["station_name"].str.strip().str.title()
    df["train_type"] = df["train_type"].str.strip().str.upper()
    df["delay_minutes"] = pd.to_numeric(df["delay_minutes"], errors="coerce").fillna(0.0)
    df["is_cancelled"] = df["is_cancelled"].astype(bool)

    df = _add_time_features(df)
    return df[_CANONICAL_COLUMNS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 1 — HuggingFace  piebro/deutsche-bahn-data
# ---------------------------------------------------------------------------

def load_huggingface(max_rows: int = 200_000) -> pd.DataFrame:
    """Stream rows from the HuggingFace dataset and return a DataFrame."""
    print(f"[HuggingFace] Streaming up to {max_rows:,} rows …")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("  datasets package not installed — skipping HuggingFace source.")
        return pd.DataFrame()

    ds = load_dataset(
        "piebro/deutsche-bahn-data",
        split="train",
        streaming=True,
    )

    records: list[dict] = []
    for row in ds:
        # HF column → canonical name mapping
        records.append(
            {
                "station_name": row.get("station_name") or row.get("xml_station_name"),
                "train_type": row.get("train_type"),
                "planned_departure": row.get("departure_planned_time") or row.get("time"),
                "delay_minutes": row.get("delay_in_min", 0),
                "is_cancelled": row.get("is_canceled", False),
            }
        )
        if len(records) >= max_rows:
            break

    print(f"  Downloaded {len(records):,} raw rows.")
    df = pd.DataFrame(records)
    df = _clean(df, source="huggingface")
    print(f"  After cleaning: {len(df):,} rows | train types: {sorted(df['train_type'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Source 2 — Project-1 PostgreSQL  db_delays.departures
# ---------------------------------------------------------------------------

def load_postgres() -> pd.DataFrame:
    """Pull all rows from the Project-1 departures table."""
    print("[PostgreSQL] Connecting to db_delays …")
    try:
        from sqlalchemy import create_engine, text  # type: ignore
    except ImportError:
        print("  sqlalchemy not installed — skipping PostgreSQL source.")
        return pd.DataFrame()

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "db_delays")
    user = os.getenv("DB_USER", "pipeline_user")
    password = os.getenv("DB_PASSWORD", "pipeline_pass")
    url = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"

    try:
        engine = create_engine(url)
        query = text(
            """
            SELECT
                station_name,
                train_type,
                planned_departure,
                delay_minutes,
                is_cancelled
            FROM departures
            WHERE planned_departure IS NOT NULL
              AND train_type IS NOT NULL
            ORDER BY planned_departure DESC
            """
        )
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        print(f"  Fetched {len(df):,} rows from departures.")
    except Exception as exc:
        print(f"  Could not connect to PostgreSQL: {exc}")
        print("  Skipping PostgreSQL source — use --no-postgres to suppress this warning.")
        return pd.DataFrame()

    df = _clean(df, source="postgres")
    print(f"  After cleaning: {len(df):,} rows | train types: {sorted(df['train_type'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(hf_rows: int, use_postgres: bool) -> None:
    frames: list[pd.DataFrame] = []

    hf_df = load_huggingface(max_rows=hf_rows)
    if not hf_df.empty:
        frames.append(hf_df)

    if use_postgres:
        pg_df = load_postgres()
        if not pg_df.empty:
            frames.append(pg_df)

    if not frames:
        print("ERROR: No data collected from any source. Exiting.")
        sys.exit(1)

    combined = pd.concat(frames, ignore_index=True)

    # De-duplicate: same station + departure time + train type
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=["station_name", "train_type", "planned_departure"]
    )
    dropped = before - len(combined)
    if dropped:
        print(f"[Dedup] Removed {dropped:,} duplicate rows.")

    combined = combined.sort_values("planned_departure").reset_index(drop=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Total rows:       {len(combined):,}")
    print(f"  Date range:       {combined['planned_departure'].min().date()}  →  "
          f"{combined['planned_departure'].max().date()}")
    print(f"  Delayed (>5 min): {combined['is_delayed'].sum():,}  "
          f"({combined['is_delayed'].mean() * 100:.1f}%)")
    print(f"  Cancelled:        {combined['is_cancelled'].sum():,}")
    print(f"  Sources:          {combined['source'].value_counts().to_dict()}")
    print(f"  Train types:      {sorted(combined['train_type'].unique())}")
    print(f"  Top 5 stations:")
    top5 = combined["station_name"].value_counts().head(5)
    for station, count in top5.items():
        print(f"    {station:<35} {count:,}")
    print("=" * 60)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1_048_576:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect DB delay training data.")
    parser.add_argument(
        "--hf-rows",
        type=int,
        default=200_000,
        help="Maximum rows to stream from HuggingFace (default: 200 000)",
    )
    parser.add_argument(
        "--no-postgres",
        action="store_true",
        help="Skip the PostgreSQL source even if .env is configured",
    )
    args = parser.parse_args()
    main(hf_rows=args.hf_rows, use_postgres=not args.no_postgres)
