import pandas as pd
import pytest

from otp.transform import add_derived_columns, dbm_to_mw, summarize_by_device


@pytest.mark.parametrize("dbm, mw", [(0, 1.0), (10, 10.0), (-10, 0.1), (3, 1.995)])
def test_dbm_to_mw(dbm, mw):
    assert dbm_to_mw(dbm) == pytest.approx(mw, rel=1e-3)


def test_in_spec_flag_uses_spec_limits():
    df = pd.DataFrame({
        "insertion_loss_db": [2.0, 3.5, 2.0],
        "output_power_dbm": [0.0, 0.0, -5.0],
    })
    out = add_derived_columns(df)
    assert out["in_spec"].tolist() == [True, False, False]


def test_device_fails_if_any_reading_is_out_of_spec():
    df = pd.DataFrame({
        "device_id": ["D1", "D1", "D2", "D2"],
        "station": ["S", "S", "S", "S"],
        "insertion_loss_db": [2.0, 3.4, 2.1, 2.2],
        "output_power_dbm": [0.0, 0.0, 0.0, 0.0],
    })
    summary = summarize_by_device(add_derived_columns(df)).set_index("device_id")
    assert not summary.loc["D1", "passed"]
    assert summary.loc["D2", "passed"]
    assert summary.loc["D1", "max_loss_db"] == 3.4
