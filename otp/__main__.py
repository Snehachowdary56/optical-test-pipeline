"""Command-line entry point.

Examples:
    python -m otp --generate              # make sample data, then run the pipeline
    python -m otp --raw-dir my_data/      # run on your own CSV files
"""

import argparse
from pathlib import Path

from otp import generate, pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Optical component test data pipeline")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Folder of raw CSV files")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Where to write the report and database")
    parser.add_argument("--generate", action="store_true", help="Generate sample data into --raw-dir first")
    parser.add_argument("--devices", type=int, default=200, help="Number of devices to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generated data")
    args = parser.parse_args()

    if args.generate:
        df = generate.generate_measurements(args.devices, args.seed)
        paths = generate.write_raw_csvs(df, args.raw_dir)
        print(f"Generated {len(df)} raw rows into {len(paths)} CSV files in {args.raw_dir}/")

    result = pipeline.run(args.raw_dir, args.output_dir)
    c = result.cleaning
    print(f"Cleaned: {c.rows_in} rows in -> {c.rows_out} rows out "
          f"({c.duplicates_removed} duplicates, {c.missing_removed} missing, {c.out_of_range_removed} out of range)")
    print(f"Devices: {len(result.device_summary)} | Yield: {result.yield_value * 100:.1f}%")
    print(f"Loss vs temperature: {result.regression.slope:.5f} dB/°C (R² = {result.regression.r_squared:.3f})")
    print(f"Database: {result.db_path}")
    print(f"Report:   {result.report_path}  <- open this in your browser")


if __name__ == "__main__":
    main()
