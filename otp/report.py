import base64
import html
import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw charts without needing a screen
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from otp import SPEC_MAX_INSERTION_LOSS_DB
from otp.clean import CleaningReport
from otp.stats import Regression


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _loss_histogram(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["insertion_loss_db"], bins=40, color="#3b6ea8", edgecolor="white")
    ax.axvline(SPEC_MAX_INSERTION_LOSS_DB, color="#c0392b", linestyle="--", label="Spec limit")
    ax.set_xlabel("Insertion loss (dB)")
    ax.set_ylabel("Measurements")
    ax.set_title("Insertion loss distribution")
    ax.legend()
    return _fig_to_base64(fig)


def _loss_vs_temperature(df: pd.DataFrame, reg: Regression) -> str:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    jitter = np.random.default_rng(0).uniform(-2, 2, len(df))
    ax.scatter(df["temperature_c"] + jitter, df["insertion_loss_db"], s=4, alpha=0.25, color="#3b6ea8")
    xs = np.array([df["temperature_c"].min(), df["temperature_c"].max()])
    ax.plot(xs, reg.slope * xs + reg.intercept, color="#c0392b",
            label=f"Fit: {reg.slope:.4f} dB/°C (R² = {reg.r_squared:.2f})")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Insertion loss (dB)")
    ax.set_title("Insertion loss vs temperature")
    ax.legend()
    return _fig_to_base64(fig)


def _yield_by_station(device_summary: pd.DataFrame) -> str:
    by_station = device_summary.groupby("station")["passed"].mean().sort_index() * 100
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(by_station.index, by_station.values, color="#3b6ea8")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Yield (%)")
    ax.set_title("Device yield by test station", pad=12)
    for i, v in enumerate(by_station.values):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center")
    return _fig_to_base64(fig)


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="data")


def write_report(
    path: Path,
    measurements: pd.DataFrame,
    device_summary: pd.DataFrame,
    by_temperature: pd.DataFrame,
    loss_stats: dict,
    reg: Regression,
    yield_value: float,
    cleaning: CleaningReport,
    station_table: pd.DataFrame,
) -> Path:
    failing = device_summary[~device_summary["passed"]].sort_values("max_loss_db", ascending=False).head(10)
    stats_table = pd.DataFrame([loss_stats])

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Optical Component Validation Report</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; max-width: 960px; margin: 32px auto; color: #222; padding: 0 16px; }}
  h1 {{ margin-bottom: 4px; }} .muted {{ color: #666; }}
  .kpis {{ display: flex; gap: 16px; margin: 20px 0; flex-wrap: wrap; }}
  .kpi {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px 18px; min-width: 150px; }}
  .kpi b {{ display: block; font-size: 24px; }}
  table.data {{ border-collapse: collapse; margin: 8px 0 24px; font-size: 14px; }}
  table.data th, table.data td {{ border-bottom: 1px solid #e5e5e5; padding: 6px 10px; text-align: right; }}
  table.data th {{ background: #f5f7fa; }}
  img {{ max-width: 100%; margin: 8px 0 20px; }}
</style></head><body>
<h1>Optical Component Validation Report</h1>
<p class="muted">Generated {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M"))} · Spec: insertion loss ≤ {SPEC_MAX_INSERTION_LOSS_DB} dB</p>

<div class="kpis">
  <div class="kpi">Devices tested<b>{len(device_summary)}</b></div>
  <div class="kpi">Yield<b>{yield_value * 100:.1f}%</b></div>
  <div class="kpi">Mean loss<b>{loss_stats["mean"]:.3f} dB</b></div>
  <div class="kpi">Temp. coefficient<b>{reg.slope * 1000:.2f} mdB/°C</b></div>
</div>

<h2>Data quality</h2>
<p>{cleaning.rows_in} raw rows → {cleaning.rows_out} clean rows.
Removed {cleaning.duplicates_removed} duplicates, {cleaning.missing_removed} rows with missing values,
and {cleaning.out_of_range_removed} physically impossible readings.</p>

<h2>Insertion loss statistics</h2>
{_table(stats_table)}
<img src="data:image/png;base64,{_loss_histogram(measurements)}" alt="Insertion loss histogram">

<h2>Temperature dependence</h2>
<p>Linear fit: loss = {reg.slope:.5f} × T + {reg.intercept:.3f} dB (R² = {reg.r_squared:.3f}).</p>
{_table(by_temperature)}
<img src="data:image/png;base64,{_loss_vs_temperature(measurements, reg)}" alt="Loss vs temperature">

<h2>Yield by station</h2>
{_table(station_table)}
<img src="data:image/png;base64,{_yield_by_station(device_summary)}" alt="Yield by station">

<h2>Worst failing devices</h2>
{_table(failing) if not failing.empty else "<p>No failing devices.</p>"}
</body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path
