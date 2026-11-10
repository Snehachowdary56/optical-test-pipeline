import numpy as np
import pandas as pd

from otp.clean import clean


def _row(**overrides):
    row = {
        "device_id": "DEV-0001", "station": "STN-A", "temperature_c": 25, "wavelength_nm": 1550,
        "output_power_dbm": 0.1, "insertion_loss_db": 2.0,
    }
    row.update(overrides)
    return row


def test_removes_exact_duplicates():
    df = pd.DataFrame([_row(), _row(), _row(wavelength_nm=1555)])
    cleaned, report = clean(df)
    assert len(cleaned) == 2
    assert report.duplicates_removed == 1


def test_removes_rows_with_missing_values():
    df = pd.DataFrame([_row(), _row(wavelength_nm=1555, insertion_loss_db=np.nan)])
    cleaned, report = clean(df)
    assert len(cleaned) == 1
    assert report.missing_removed == 1


def test_removes_instrument_glitches():
    df = pd.DataFrame([
        _row(),
        _row(wavelength_nm=1555, output_power_dbm=-999.0),  # sentinel value
        _row(wavelength_nm=1560, insertion_loss_db=85.0),   # impossible spike
    ])
    cleaned, report = clean(df)
    assert len(cleaned) == 1
    assert report.out_of_range_removed == 2


def test_report_counts_add_up():
    df = pd.DataFrame([_row(), _row(), _row(wavelength_nm=1555, insertion_loss_db=np.nan)])
    cleaned, r = clean(df)
    assert r.rows_in - r.duplicates_removed - r.missing_removed - r.out_of_range_removed == r.rows_out
    assert r.rows_out == len(cleaned)
