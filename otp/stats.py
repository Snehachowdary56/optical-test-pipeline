from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Regression:
    slope: float
    intercept: float
    r_squared: float


def describe(series: pd.Series) -> dict:
    """Core descriptive statistics for one measurement."""
    s = series.dropna()
    return {
        "count": int(s.size),
        "mean": round(float(s.mean()), 4),
        "std": round(float(s.std(ddof=1)), 4) if s.size > 1 else 0.0,
        "min": round(float(s.min()), 4),
        "p05": round(float(s.quantile(0.05)), 4),
        "median": round(float(s.median()), 4),
        "p95": round(float(s.quantile(0.95)), 4),
        "max": round(float(s.max()), 4),
    }


def yield_rate(device_summary: pd.DataFrame) -> float:
    """Share of devices that passed every test, between 0 and 1."""
    if device_summary.empty:
        return 0.0
    return round(float(device_summary["passed"].mean()), 4)


def linear_regression(x, y) -> Regression:
    """Least-squares straight line y = slope * x + intercept, with R-squared."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.all(x == x[0]):
        raise ValueError("Need at least two distinct x values for a regression.")

    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return Regression(round(float(slope), 6), round(float(intercept), 4), round(r_squared, 4))


def loss_vs_temperature(df: pd.DataFrame) -> Regression:
    """How insertion loss changes per degree C across all devices."""
    return linear_regression(df["temperature_c"], df["insertion_loss_db"])
