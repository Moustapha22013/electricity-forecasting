"""Loader: missing tokens, day-first dates, schema validation."""
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import NUMERIC_COLUMNS, load_raw

HEADER = "Date;Time;" + ";".join(NUMERIC_COLUMNS)


def _write(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "raw.txt"
    p.write_bytes(("\r\n".join([HEADER, *lines]) + "\r\n").encode())   # CRLF like the real file
    return p


def test_missing_tokens_and_dayfirst(tmp_path):
    p = _write(tmp_path, ["01/02/2007;00:00:00;1.0;0.1;240.0;4.2;0.000;1.000;17.000",
                          "01/02/2007;00:01:00;?;?;?;?;?;?;"])
    df = load_raw(p)
    assert df["timestamp"].iloc[0].month == 2 and df["timestamp"].iloc[0].day == 1   # 01/02 = 1 Feb
    assert df.loc[1, NUMERIC_COLUMNS].isna().all()          # '?' and empty field both -> NaN
    assert df["Global_active_power"].dtype == np.float64


def test_schema_error(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("Date;Time;Voltage\n01/01/2007;00:00:00;240\n")
    with pytest.raises(ValueError):
        load_raw(p)


def test_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.txt")
