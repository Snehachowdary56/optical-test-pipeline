"""Optical Component Test Data Pipeline.

Loads raw test measurements for optical components, cleans and transforms them,
stores them in DuckDB, computes validation statistics, and writes an HTML report.
"""

# Spec limits used for pass/fail. A device passes if every measurement is within limits.
SPEC_MAX_INSERTION_LOSS_DB = 3.0
SPEC_MIN_OUTPUT_POWER_DBM = -3.0

# Physically plausible ranges. Readings outside these are treated as instrument errors.
VALID_INSERTION_LOSS_DB = (0.0, 20.0)
VALID_OUTPUT_POWER_DBM = (-40.0, 10.0)

REQUIRED_COLUMNS = [
    "device_id", "station", "temperature_c", "wavelength_nm",
    "output_power_dbm", "insertion_loss_db",
]
