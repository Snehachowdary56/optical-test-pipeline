import numpy as np
import pandas as pd

from otp import SPEC_MAX_INSERTION_LOSS_DB, SPEC_MIN_OUTPUT_POWER_DBM


def dbm_to_mw(dbm):
    """Converts optical power from dBm (logarithmic) to milliwatts (linear): P_mW = 10^(dBm/10)."""
    return np.power(10.0, np.asarray(dbm, dtype=float) / 10.0)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["output_power_mw"] = dbm_to_mw(df["output_power_dbm"]).round(4)
    df["in_spec"] = (
        (df["insertion_loss_db"] <= SPEC_MAX_INSERTION_LOSS_DB)
        & (df["output_power_dbm"] >= SPEC_MIN_OUTPUT_POWER_DBM)
    )
    return df


def summarize_by_device(df: pd.DataFrame) -> pd.DataFrame:
    """One row per device: worst-case loss, average power, and whether every reading was in spec."""
    summary = (
        df.groupby(["device_id", "station"], as_index=False)
        .agg(
            measurements=("insertion_loss_db", "size"),
            mean_loss_db=("insertion_loss_db", "mean"),
            max_loss_db=("insertion_loss_db", "max"),
            mean_power_dbm=("output_power_dbm", "mean"),
            min_power_dbm=("output_power_dbm", "min"),
            passed=("in_spec", "all"),
        )
    )
    return summary.round({"mean_loss_db": 3, "max_loss_db": 3, "mean_power_dbm": 3, "min_power_dbm": 3})


def summarize_by_temperature(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("temperature_c", as_index=False)
        .agg(
            mean_loss_db=("insertion_loss_db", "mean"),
            std_loss_db=("insertion_loss_db", "std"),
            in_spec_rate=("in_spec", "mean"),
        )
        .round(4)
    )
