"""End-to-end test: generate data, run the whole pipeline, and check the outputs."""

import pytest

from otp import generate, pipeline, storage
from otp.generate import TEMP_COEFF_DB_PER_C


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    base = tmp_path_factory.mktemp("run")
    df = generate.generate_measurements(n_devices=60, seed=7)
    generate.write_raw_csvs(df, base / "raw")
    return pipeline.run(base / "raw", base / "out")


def test_pipeline_removes_injected_problems(result):
    c = result.cleaning
    assert c.duplicates_removed > 0
    assert c.missing_removed > 0
    assert c.out_of_range_removed > 0
    assert result.measurements["insertion_loss_db"].max() < 20


def test_pipeline_recovers_temperature_coefficient(result):
    # The generator builds in a known coefficient; the regression should find it.
    assert result.regression.slope == pytest.approx(TEMP_COEFF_DB_PER_C, abs=0.001)


def test_pipeline_writes_database_and_report(result):
    assert result.report_path.exists()
    assert "Validation Report" in result.report_path.read_text(encoding="utf-8")
    counts = storage.query(result.db_path, "SELECT COUNT(*) AS n FROM device_summary")
    assert counts["n"][0] == 60
