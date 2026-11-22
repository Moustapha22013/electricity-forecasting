"""Download and load the raw UCI household power consumption file.

The loader does NOT clean, sort or impute anything: it only parses. Every
transformation happens later, in `preprocessing.py`, so that the raw state can
always be audited.
"""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

NUMERIC_COLUMNS: list[str] = [
    "Global_active_power",    # kW, minute-averaged
    "Global_reactive_power",  # kVAR, minute-averaged
    "Voltage",                # V, minute-averaged
    "Global_intensity",       # A, minute-averaged
    "Sub_metering_1",         # Wh of active energy during the minute (kitchen)
    "Sub_metering_2",         # Wh (laundry room)
    "Sub_metering_3",         # Wh (water heater + air conditioner)
]
# Verified on the raw file: in a missing row the first six measures are "?" and
# Sub_metering_3 is an empty field. Both tokens encode the same outage.
MISSING_TOKENS: list[str] = ["?", ""]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_raw(url: str, raw_dir: str | Path, raw_file: str, expected_sha256: str | None = None) -> Path:
    """Download and unzip the dataset if it is not already present. Verifies the archive checksum."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / raw_file
    if target.exists():
        return target
    archive = raw_dir / "household.zip"
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    if expected_sha256 and _sha256(archive) != expected_sha256:
        raise ValueError(f"Checksum mismatch for {archive}: the downloaded file is not the expected version.")
    with zipfile.ZipFile(archive) as z:
        z.extract(raw_file, raw_dir)
    return target


def load_raw(path: str | Path) -> pd.DataFrame:
    """Parse the raw file into a DataFrame with a `timestamp` column, in file order.

    * `?` and empty fields are the dataset's missing-value tokens -> NaN.
    * Date is day-first (dd/mm/yyyy); an explicit format avoids silent month/day swaps.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run the download step first.")
    df = pd.read_csv(path, sep=";", na_values=MISSING_TOKENS, keep_default_na=False,
                     dtype={c: "float64" for c in NUMERIC_COLUMNS} | {"Date": str, "Time": str})
    missing_cols = set(NUMERIC_COLUMNS + ["Date", "Time"]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing_cols)}")
    df.insert(0, "timestamp", pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"))
    return df.drop(columns=["Date", "Time"])
