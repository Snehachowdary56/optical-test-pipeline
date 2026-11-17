"""Tools the chatbot can call. Each one runs SQL against the DuckDB database and returns plain data."""

from __future__ import annotations

from pathlib import Path

import duckdb

from otp import SPEC_MAX_INSERTION_LOSS_DB, SPEC_MIN_OUTPUT_POWER_DBM

METRICS = {"insertion_loss_db": "insertion loss (dB)", "output_power_dbm": "output power (dBm)"}


class DataTools:
    def __init__(self, db_path: Path, cleaning: dict | None = None):
        self.db_path = Path(db_path)
        self.cleaning = cleaning or {}

    def _q(self, sql: str, params: list | None = None) -> list[dict]:
        with duckdb.connect(str(self.db_path), read_only=True) as con:
            cur = con.execute(sql, params or [])
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    @staticmethod
    def _where(station: str | None, temperature_c: int | None) -> tuple[str, list]:
        clauses, params = [], []
        if station:
            clauses.append("station = ?")
            params.append(station)
        if temperature_c is not None:
            clauses.append("temperature_c = ?")
            params.append(int(temperature_c))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    # ---- tools -------------------------------------------------------------------------
    def overview(self) -> dict:
        r = self._q("""SELECT COUNT(*) AS measurements, COUNT(DISTINCT device_id) AS devices,
                              COUNT(DISTINCT station) AS stations, COUNT(DISTINCT temperature_c) AS temperatures,
                              COUNT(DISTINCT wavelength_nm) AS wavelengths,
                              MIN(wavelength_nm) AS min_wavelength_nm, MAX(wavelength_nm) AS max_wavelength_nm
                       FROM measurements""")[0]
        r["station_names"] = [x["station"] for x in self._q("SELECT DISTINCT station FROM measurements ORDER BY 1")]
        r["temperature_list_c"] = [x["t"] for x in self._q("SELECT DISTINCT temperature_c AS t FROM measurements ORDER BY 1")]
        return r

    def device_yield(self, station: str | None = None) -> dict:
        where, params = self._where(station, None)
        r = self._q(f"""SELECT COUNT(*) AS devices, CAST(SUM(CASE WHEN passed THEN 1 ELSE 0 END) AS INTEGER) AS passed
                        FROM device_summary{where}""", params)[0]
        r["failed"] = r["devices"] - (r["passed"] or 0)
        r["yield_pct"] = round(100 * r["passed"] / r["devices"], 1) if r["devices"] else 0.0
        r["station"] = station
        return r

    def yield_by_station(self) -> dict:
        rows = self._q("""SELECT station, COUNT(*) AS devices,
                                 CAST(SUM(CASE WHEN passed THEN 0 ELSE 1 END) AS INTEGER) AS failed,
                                 ROUND(100.0 * AVG(CASE WHEN passed THEN 1 ELSE 0 END), 1) AS yield_pct
                          FROM device_summary GROUP BY station ORDER BY station""")
        return {"stations": rows}

    def in_spec_rate(self, temperature_c: int | None = None, station: str | None = None) -> dict:
        where, params = self._where(station, temperature_c)
        r = self._q(f"""SELECT COUNT(*) AS measurements,
                               ROUND(100.0 * AVG(CASE WHEN in_spec THEN 1 ELSE 0 END), 1) AS in_spec_pct
                        FROM measurements{where}""", params)[0]
        r.update(temperature_c=temperature_c, station=station)
        return r

    def metric_stats(self, metric: str = "insertion_loss_db", temperature_c: int | None = None,
                     station: str | None = None) -> dict:
        if metric not in METRICS:
            raise ValueError(f"unknown metric {metric}")
        where, params = self._where(station, temperature_c)
        r = self._q(f"""SELECT COUNT(*) AS n, ROUND(AVG({metric}), 3) AS mean, ROUND(STDDEV_SAMP({metric}), 3) AS std,
                               ROUND(MIN({metric}), 3) AS min, ROUND(MEDIAN({metric}), 3) AS median,
                               ROUND(QUANTILE_CONT({metric}, 0.95), 3) AS p95, ROUND(MAX({metric}), 3) AS max
                        FROM measurements{where}""", params)[0]
        r.update(metric=metric, temperature_c=temperature_c, station=station)
        return r

    def worst_devices(self, n: int = 5, station: str | None = None) -> dict:
        where, params = self._where(station, None)
        rows = self._q(f"""SELECT device_id, station, max_loss_db, min_power_dbm, passed
                           FROM device_summary{where} ORDER BY max_loss_db DESC, device_id LIMIT ?""",
                       params + [int(n)])
        return {"devices": rows, "n": int(n), "station": station}

    def failed_devices(self, station: str | None = None) -> dict:
        where, params = self._where(station, None)
        where = (where + " AND " if where else " WHERE ") + "NOT passed"
        rows = self._q(f"SELECT device_id, station, max_loss_db, min_power_dbm FROM device_summary{where} ORDER BY device_id",
                       params)
        return {"count": len(rows), "devices": rows, "station": station}

    def device_detail(self, device_id: str) -> dict:
        rows = self._q("SELECT * FROM device_summary WHERE device_id = ?", [device_id])
        if not rows:
            return {"device_id": device_id, "found": False}
        r = rows[0]
        r["found"] = True
        r["failing_readings"] = self._q(
            """SELECT temperature_c, wavelength_nm, insertion_loss_db, output_power_dbm FROM measurements
               WHERE device_id = ? AND NOT in_spec ORDER BY temperature_c, wavelength_nm""", [device_id])
        r["failing_count"] = len(r["failing_readings"])
        return r

    def temperature_effect(self) -> dict:
        r = self._q("""SELECT REGR_SLOPE(insertion_loss_db, temperature_c) AS slope,
                              REGR_INTERCEPT(insertion_loss_db, temperature_c) AS intercept,
                              REGR_R2(insertion_loss_db, temperature_c) AS r2 FROM measurements""")[0]
        by_t = self._q("""SELECT temperature_c, ROUND(AVG(insertion_loss_db), 3) AS mean_loss_db
                          FROM measurements GROUP BY 1 ORDER BY 1""")
        return {"slope_db_per_c": round(r["slope"], 5), "intercept_db": round(r["intercept"], 3),
                "r_squared": round(r["r2"], 3), "by_temperature": by_t}

    def spec_limits(self) -> dict:
        return {"max_insertion_loss_db": SPEC_MAX_INSERTION_LOSS_DB, "min_output_power_dbm": SPEC_MIN_OUTPUT_POWER_DBM}

    def cleaning_summary(self) -> dict:
        return dict(self.cleaning)


# Descriptions used for LLM tool calling (JSON Schema).
TOOL_SPECS = [
    {"name": "overview", "description": "Counts of devices, measurements, stations, temperatures and wavelengths.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "device_yield", "description": "Device yield: how many devices passed every test, optionally for one station.",
     "input_schema": {"type": "object", "properties": {"station": {"type": "string", "enum": ["STN-A", "STN-B", "STN-C"]}}}},
    {"name": "yield_by_station", "description": "Yield and failed devices for every station.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "in_spec_rate", "description": "Share of individual measurements within spec, optionally at a temperature and/or station.",
     "input_schema": {"type": "object", "properties": {"temperature_c": {"type": "integer"}, "station": {"type": "string"}}}},
    {"name": "metric_stats", "description": "Statistics (mean, std, min, median, p95, max) of insertion_loss_db or output_power_dbm.",
     "input_schema": {"type": "object", "properties": {
         "metric": {"type": "string", "enum": list(METRICS)}, "temperature_c": {"type": "integer"}, "station": {"type": "string"}},
         "required": ["metric"]}},
    {"name": "worst_devices", "description": "Devices with the highest worst-case insertion loss.",
     "input_schema": {"type": "object", "properties": {"n": {"type": "integer"}, "station": {"type": "string"}}}},
    {"name": "failed_devices", "description": "List of devices that failed at least one test.",
     "input_schema": {"type": "object", "properties": {"station": {"type": "string"}}}},
    {"name": "device_detail", "description": "Summary and failing readings for one device, e.g. DEV-0042.",
     "input_schema": {"type": "object", "properties": {"device_id": {"type": "string"}}, "required": ["device_id"]}},
    {"name": "temperature_effect", "description": "Linear regression of insertion loss vs temperature (slope dB/°C, R²).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "spec_limits", "description": "Pass/fail spec limits.", "input_schema": {"type": "object", "properties": {}}},
    {"name": "cleaning_summary", "description": "How many raw rows were removed during cleaning and why.",
     "input_schema": {"type": "object", "properties": {}}},
]
