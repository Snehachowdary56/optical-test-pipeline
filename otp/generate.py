"""Generates realistic, slightly messy test data so the pipeline has something to clean.

Each device is measured at several temperatures and wavelengths on one of a few test stations.
Insertion loss rises with temperature, and a small share of devices are out of spec.
The generator also injects the kinds of problems real lab data has: missing values,
duplicate rows and instrument glitches.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TEMPERATURES_C = [-5, 25, 75]
WAVELENGTHS_NM = list(range(1530, 1566, 5))  # C-band: 1530 to 1565 nm
STATIONS = ["STN-A", "STN-B", "STN-C"]
TEMP_COEFF_DB_PER_C = 0.008  # insertion loss increase per degree C


def generate_measurements(n_devices: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_devices):
        device_id = f"DEV-{i + 1:04d}"
        station = STATIONS[i % len(STATIONS)]
        base_loss = rng.normal(2.0, 0.3)  # each device has its own typical loss
        base_power = rng.normal(0.0, 0.8)

        for temp in TEMPERATURES_C:
            for wl in WAVELENGTHS_NM:
                # Loss is slightly higher at the edges of the band.
                wl_effect = 0.0004 * (wl - 1547.5) ** 2 / 10
                loss = base_loss + TEMP_COEFF_DB_PER_C * (temp - 25) + wl_effect + rng.normal(0, 0.05)
                power = base_power - (loss - 2.0) + rng.normal(0, 0.1)
                rows.append((device_id, station, temp, wl, round(power, 3), round(loss, 3)))

    df = pd.DataFrame(rows, columns=[
        "device_id", "station", "temperature_c", "wavelength_nm",
        "output_power_dbm", "insertion_loss_db",
    ])
    return _inject_problems(df, rng)


def _inject_problems(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # About 1% missing readings.
    missing = rng.choice(n, size=max(1, n // 100), replace=False)
    df.loc[missing[: len(missing) // 2], "insertion_loss_db"] = np.nan
    df.loc[missing[len(missing) // 2:], "output_power_dbm"] = np.nan

    # About 0.5% instrument glitches: a sentinel value of -999 or an impossible spike.
    glitches = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[glitches[::2], "output_power_dbm"] = -999.0
    df.loc[glitches[1::2], "insertion_loss_db"] = 85.0

    # About 1% duplicate rows (the same measurement logged twice).
    dupes = df.sample(n=max(1, n // 100), random_state=int(rng.integers(0, 10_000)))
    return pd.concat([df, dupes], ignore_index=True)


def write_raw_csvs(df: pd.DataFrame, raw_dir: Path) -> list[Path]:
    """Writes one CSV per test station, the way separate lab stations would export results."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for station, group in df.groupby("station"):
        path = raw_dir / f"{station.lower()}_results.csv"
        group.to_csv(path, index=False)
        paths.append(path)
    return paths
