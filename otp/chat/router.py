"""Offline intent router: turns a question into a tool call with arguments, using simple rules.

It also remembers the last tool call so follow-ups like "what about at 75C?" or "and station B?" work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

STATION_RE = re.compile(r"\b(?:stn[-\s]?|station\s+)([abc])\b", re.I)
TEMP_RE = re.compile(r"(-?\d+)\s*(?:°\s*c|deg(?:rees?)?(?:\s*c(?:elsius)?)?|c\b)", re.I)
DEVICE_RE = re.compile(r"\bdev[-\s]?(\d{1,4})\b", re.I)
TOP_N_RE = re.compile(r"\b(?:top|worst)\s+(\d{1,2})\b", re.I)

DOMAIN_WORDS = ("yield", "pass", "fail", "device", "loss", "power", "spec", "station", "temperature", "temp",
                "wavelength", "measurement", "reading", "clean", "duplicate", "missing", "glitch", "outlier",
                "dbm", "db", "worst", "regression", "slope", "test", "data")


@dataclass
class ToolCall:
    tool: str
    args: dict = field(default_factory=dict)


def _station(q: str) -> str | None:
    m = STATION_RE.search(q)
    return f"STN-{m.group(1).upper()}" if m else None


def _temperature(q: str) -> int | None:
    m = TEMP_RE.search(q)
    return int(m.group(1)) if m else None


def _metric(q: str) -> str:
    return "output_power_dbm" if re.search(r"\bpower\b|\bdbm\b", q, re.I) else "insertion_loss_db"


def route(question: str, previous: ToolCall | None = None) -> ToolCall:
    q = question.strip().lower()
    station, temp = _station(q), _temperature(q)

    if re.fullmatch(r"(hi|hello|hey|help|what can you do\??|\?)", q):
        return ToolCall("help")

    dev = DEVICE_RE.search(q)
    if dev:
        return ToolCall("device_detail", {"device_id": f"DEV-{int(dev.group(1)):04d}"})

    # Follow-up questions reuse the previous tool with new filters.
    if previous and re.match(r"^(what about|how about|and|same for|now)\b", q) and (station or temp is not None):
        args = dict(previous.args)
        if station:
            args["station"] = station
        if temp is not None and previous.tool in ("metric_stats", "in_spec_rate"):
            args["temperature_c"] = temp
        tool = previous.tool
        if tool == "yield_by_station" and station:
            tool, args = "device_yield", {"station": station}
        return ToolCall(tool, args)

    if re.search(r"spec limit|limits?\b|threshold|pass criteria|what (?:is|counts as) (?:a )?pass", q):
        return ToolCall("spec_limits")
    if re.search(r"clean|duplicate|missing|glitch|removed|dropped|bad (?:rows|data)|sentinel|-999", q):
        return ToolCall("cleaning_summary")
    if re.search(r"temperature (?:effect|affect|impact|depend)|vs\.? temperature|versus temperature|per degree|"
                 r"slope|regression|r2|r²|r-squared|does temperature|how does temperature", q):
        return ToolCall("temperature_effect")
    if re.search(r"worst|highest (?:insertion )?loss|top \d+", q):
        n = TOP_N_RE.search(q)
        return ToolCall("worst_devices", {"n": int(n.group(1)) if n else 5, **({"station": station} if station else {})})
    if re.search(r"which devices fail|list (?:the )?fail|failed devices|failing devices|devices (?:that|which) failed", q):
        return ToolCall("failed_devices", {"station": station} if station else {})
    if re.search(r"(each|every|per|by|all|compare) (?:the )?stations?|stations? compar|which station", q):
        return ToolCall("yield_by_station")
    if re.search(r"in[- ]spec|within spec|out of spec|measurements? pass|readings? pass", q) or (
            temp is not None and re.search(r"pass|yield|fail", q)):
        return ToolCall("in_spec_rate", {k: v for k, v in (("temperature_c", temp), ("station", station)) if v is not None})
    if re.search(r"yield|pass rate|how many (?:devices )?(?:passed|failed)|fail(?:ed|ure)? rate|percent.*pass", q):
        return ToolCall("device_yield", {"station": station} if station else {})
    if re.search(r"average|mean|median|std|standard deviation|spread|p95|percentile|max(?:imum)?|min(?:imum)?|"
                 r"statistic|distribution|typical", q) and re.search(r"loss|power|db", q):
        return ToolCall("metric_stats", {"metric": _metric(q),
                                         **({"temperature_c": temp} if temp is not None else {}),
                                         **({"station": station} if station else {})})
    if re.search(r"how many (?:devices|measurements|readings|stations|temperatures|wavelengths)|overview|summary|"
                 r"what data|which (?:temperatures|wavelengths)|dataset", q):
        return ToolCall("overview")

    if not any(w in q for w in DOMAIN_WORDS):
        return ToolCall("out_of_scope")
    return ToolCall("unclear")
