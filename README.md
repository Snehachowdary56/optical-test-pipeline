# Optical Component Test Data Pipeline

A Python data pipeline that turns raw test measurements from optical components into clean data, validation statistics and an HTML report for engineering review.

![CI](https://github.com/Snehachowdary56/optical-test-pipeline/actions/workflows/ci.yml/badge.svg)

## What it does

```
raw CSVs ──► load ──► clean ──► transform ──► DuckDB (SQL) ──► statistics ──► HTML report
```

1. **Load**: reads every CSV in a folder (one per test station) and checks the required columns are present.
2. **Clean**: removes duplicate rows, rows with missing values, and physically impossible readings such as `-999` sentinel values from the instrument. It reports how many rows were dropped and why.
3. **Transform**: converts optical power from dBm to mW, flags each reading against the spec limits, and summarizes results per device and per temperature.
4. **Store**: saves the clean measurements and device summaries as SQL tables in a DuckDB database, then queries them with SQL.
5. **Analyze**: computes descriptive statistics (mean, standard deviation, percentiles), yield (the share of devices that pass every test), and a linear regression of insertion loss against temperature, with R².
6. **Report**: writes a self-contained HTML report with key metrics, tables and charts.

Because there's no real lab data, `otp/generate.py` creates realistic sample data: 200 devices measured at 3 temperatures and 8 C-band wavelengths, with a built-in temperature effect and deliberately injected problems (missing values, duplicates, instrument glitches) for the pipeline to clean.

## Tech stack

Python · Pandas · NumPy · DuckDB (SQL) · Matplotlib · pytest · GitHub Actions

## Run it

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m otp --generate
```

Then open `output/report.html` in your browser.

Example output:

```
Generated 4848 raw rows into 3 CSV files in data/raw/
Cleaned: 4848 rows in -> 4728 rows out (48 duplicates, 48 missing, 24 out of range)
Devices: 200 | Yield: 98.0%
Loss vs temperature: 0.00798 dB/°C (R² = 0.478)
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--generate` | off | Create sample data first |
| `--devices` | 200 | Number of devices to generate |
| `--raw-dir` | `data/raw` | Folder of input CSVs |
| `--output-dir` | `output` | Where the report and database go |

### Run the tests

```bash
pytest
```

18 tests cover cleaning, unit conversion, spec checks, statistics, and a full end-to-end run. The end-to-end test checks that the pipeline recovers the temperature coefficient that was built into the generated data.

## Project structure

```
otp/
  generate.py   Sample data with realistic noise and injected problems
  load.py       Reads and validates raw CSVs
  clean.py      Removes duplicates, missing values and impossible readings
  transform.py  Unit conversion, spec flags, per-device and per-temperature summaries
  storage.py    DuckDB tables and SQL queries
  stats.py      Descriptive statistics, yield, linear regression
  report.py     HTML report with charts
  pipeline.py   Runs every step in order
tests/          pytest tests
```

## Design decisions

- **Each step is a separate, testable function**, so cleaning rules or statistics can change without touching the rest.
- **Bad data is counted, not silently dropped.** The report states how many rows were removed and why, so engineers can trust the numbers.
- **Pass/fail is per device, worst case.** A device fails if any single reading is out of spec, which is how validation usually works.
- **DuckDB** is used because it runs SQL analytics directly on DataFrames with no server to set up.

## Ideas for next steps

- Outlier detection with statistical rules (e.g. z-score or IQR per test condition)
- Station-to-station comparison with a hypothesis test (e.g. ANOVA)
- Process capability metrics (Cpk) against the spec limits
- Read larger datasets directly with DuckDB or Polars instead of Pandas
