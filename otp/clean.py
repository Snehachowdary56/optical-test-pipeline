from dataclasses import dataclass, field

import pandas as pd

from otp import REQUIRED_COLUMNS, VALID_INSERTION_LOSS_DB, VALID_OUTPUT_POWER_DBM


@dataclass
class CleaningReport:
    """Counts of what was removed, so engineers can see how much data was dropped and why."""
    rows_in: int = 0
    duplicates_removed: int = 0
    missing_removed: int = 0
    out_of_range_removed: int = 0
    rows_out: int = 0
    notes: list[str] = field(default_factory=list)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    report = CleaningReport(rows_in=len(df))

    # 1. Exact duplicate measurements (same device, conditions and readings).
    before = len(df)
    df = df.drop_duplicates(subset=REQUIRED_COLUMNS)
    report.duplicates_removed = before - len(df)

    # 2. Rows with any missing required value.
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    report.missing_removed = before - len(df)

    # 3. Physically impossible readings (instrument errors such as -999 sentinels).
    before = len(df)
    lo_il, hi_il = VALID_INSERTION_LOSS_DB
    lo_p, hi_p = VALID_OUTPUT_POWER_DBM
    in_range = (
        df["insertion_loss_db"].between(lo_il, hi_il)
        & df["output_power_dbm"].between(lo_p, hi_p)
    )
    df = df[in_range]
    report.out_of_range_removed = before - len(df)

    df = df.astype({"temperature_c": int, "wavelength_nm": int}).reset_index(drop=True)
    report.rows_out = len(df)
    return df, report
