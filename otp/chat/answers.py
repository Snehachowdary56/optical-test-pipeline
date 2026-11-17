"""Turns tool results into short, plain-English answers. Every number comes from the tool result."""

from __future__ import annotations

from otp.chat.tools import METRICS

HELP = ("I answer questions about the optical component test data, for example: "
        "\"What is the yield?\", \"Compare stations\", \"Average insertion loss at 75C\", "
        "\"Which devices failed?\", \"Tell me about DEV-0042\", \"How does temperature affect loss?\", "
        "\"What did cleaning remove?\" or \"What are the spec limits?\"")
OUT_OF_SCOPE = "Sorry, I can only answer questions about the optical component test data. " + HELP
UNCLEAR = ("I'm not sure what you're asking about the test data. Try asking about yield, stations, "
           "insertion loss, output power, temperature, a device ID or data cleaning.")


def _scope(station=None, temperature_c=None) -> str:
    parts = []
    if station:
        parts.append(f"at {station}")
    if temperature_c is not None:
        parts.append(f"at {temperature_c}°C")
    return (" " + " ".join(parts)) if parts else ""


def format_answer(tool: str, r: dict) -> str:
    if tool == "help":
        return HELP
    if tool == "out_of_scope":
        return OUT_OF_SCOPE
    if tool == "unclear":
        return UNCLEAR
    if tool == "overview":
        return (f"The data has {r['devices']} devices and {r['measurements']} clean measurements from "
                f"{r['stations']} stations ({', '.join(r['station_names'])}), at {r['temperatures']} temperatures "
                f"({', '.join(f'{t}°C' for t in r['temperature_list_c'])}) and {r['wavelengths']} wavelengths "
                f"({r['min_wavelength_nm']}–{r['max_wavelength_nm']} nm).")
    if tool == "device_yield":
        return (f"Yield{_scope(r.get('station'))} is {r['yield_pct']}%: {r['passed']} of {r['devices']} devices "
                f"passed every test and {r['failed']} failed.")
    if tool == "yield_by_station":
        parts = [f"{s['station']}: {s['yield_pct']}% ({s['failed']} of {s['devices']} failed)" for s in r["stations"]]
        worst = min(r["stations"], key=lambda s: s["yield_pct"])
        return "Yield by station: " + "; ".join(parts) + f". Lowest: {worst['station']}."
    if tool == "in_spec_rate":
        return (f"{r['in_spec_pct']}% of {r['measurements']} measurements{_scope(r.get('station'), r.get('temperature_c'))} "
                f"were within spec.")
    if tool == "metric_stats":
        name = METRICS[r["metric"]]
        return (f"{name[0].upper() + name[1:]}{_scope(r.get('station'), r.get('temperature_c'))} over {r['n']} measurements: "
                f"mean {r['mean']}, median {r['median']}, std {r['std']}, min {r['min']}, 95th percentile {r['p95']}, "
                f"max {r['max']}.")
    if tool == "worst_devices":
        rows = "; ".join(f"{d['device_id']} ({d['station']}) {d['max_loss_db']} dB{'' if d['passed'] else ' FAIL'}"
                         for d in r["devices"])
        return f"Top {len(r['devices'])} devices by worst-case insertion loss{_scope(r.get('station'))}: {rows}."
    if tool == "failed_devices":
        if not r["count"]:
            return f"No devices failed{_scope(r.get('station'))}."
        ids = ", ".join(d["device_id"] for d in r["devices"][:10])
        more = f" and {r['count'] - 10} more" if r["count"] > 10 else ""
        return f"{r['count']} devices failed{_scope(r.get('station'))}: {ids}{more}."
    if tool == "device_detail":
        if not r["found"]:
            return f"I couldn't find {r['device_id']} in the test data."
        n = r["failing_count"]
        status = "PASSED" if r["passed"] else f"FAILED ({n} reading{'s' if n != 1 else ''} out of spec)"
        return (f"{r['device_id']} was tested at {r['station']} ({r['measurements']} measurements) and {status}. "
                f"Worst insertion loss {r['max_loss_db']} dB, mean {r['mean_loss_db']} dB; "
                f"lowest output power {r['min_power_dbm']} dBm.")
    if tool == "temperature_effect":
        means = ", ".join(f"{t['temperature_c']}°C: {t['mean_loss_db']} dB" for t in r["by_temperature"])
        return (f"Insertion loss rises about {r['slope_db_per_c']} dB per °C (linear regression, R² = {r['r_squared']}). "
                f"Mean loss by temperature: {means}.")
    if tool == "spec_limits":
        return (f"A measurement passes if insertion loss is at most {r['max_insertion_loss_db']} dB and output power is "
                f"at least {r['min_output_power_dbm']} dBm. A device passes only if every measurement passes.")
    if tool == "cleaning_summary":
        return (f"Cleaning took {r['rows_in']} raw rows down to {r['rows_out']}: removed {r['duplicates_removed']} duplicates, "
                f"{r['missing_removed']} rows with missing values and {r['out_of_range_removed']} impossible readings "
                f"(instrument glitches such as sentinel values).")
    raise ValueError(f"no formatter for {tool}")
