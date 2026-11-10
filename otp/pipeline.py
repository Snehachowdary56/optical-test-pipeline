"""Runs the full pipeline: load -> clean -> transform -> store -> analyze -> report."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from otp import load, storage
from otp.clean import CleaningReport, clean
from otp.report import write_report
from otp.stats import Regression, describe, loss_vs_temperature, yield_rate
from otp.transform import add_derived_columns, summarize_by_device, summarize_by_temperature


@dataclass
class PipelineResult:
    measurements: pd.DataFrame
    device_summary: pd.DataFrame
    cleaning: CleaningReport
    regression: Regression
    yield_value: float
    report_path: Path
    db_path: Path


def run(raw_dir: Path, output_dir: Path) -> PipelineResult:
    raw = load.load_raw(raw_dir)
    cleaned, cleaning = clean(raw)
    measurements = add_derived_columns(cleaned)

    device_summary = summarize_by_device(measurements)
    by_temperature = summarize_by_temperature(measurements)

    db_path = output_dir / "test_results.duckdb"
    storage.save_to_duckdb(db_path, measurements, device_summary)
    station_table = storage.query(db_path, storage.FAILING_DEVICES_BY_STATION)

    loss_stats = describe(measurements["insertion_loss_db"])
    reg = loss_vs_temperature(measurements)
    yield_value = yield_rate(device_summary)

    report_path = write_report(
        output_dir / "report.html", measurements, device_summary, by_temperature,
        loss_stats, reg, yield_value, cleaning, station_table,
    )
    return PipelineResult(measurements, device_summary, cleaning, reg, yield_value, report_path, db_path)
