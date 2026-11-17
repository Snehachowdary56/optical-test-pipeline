"""Groundedness guard: an LLM answer is accepted only if every number in it appears in the tool results."""

from __future__ import annotations

import re

NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
ID_DIGITS_RE = re.compile(r"\b(?:DEV|STN)-[A-Z0-9]+\b", re.I)


def _numbers_in(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out += [float(x.replace(",", "")) for x in NUM_RE.findall(ID_DIGITS_RE.sub("", obj))]
    elif isinstance(obj, dict):
        for v in obj.values():
            out += _numbers_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out += _numbers_in(v)
    return out


def ungrounded_numbers(answer: str, tool_result) -> list[float]:
    """Numbers in the answer that are not in the tool result.

    A number is grounded if some tool value rounds to it at the precision written in the answer
    (so 27.85 matches 27.8467), or if it is a 0-1 share written as a percentage (0.98 -> 98).
    """
    allowed = _numbers_in(tool_result)
    allowed += [a * 100 for a in allowed if -1 <= a <= 1]
    bad = []
    for text in NUM_RE.findall(ID_DIGITS_RE.sub("", answer.replace("95th", ""))):
        n = float(text.replace(",", ""))
        decimals = len(text.split(".")[1]) if "." in text else 0
        if not any(abs(round(a, decimals) - n) < 1e-9 for a in allowed):
            bad.append(n)
    return bad
