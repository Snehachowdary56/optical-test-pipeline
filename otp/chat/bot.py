"""The chatbot: route the question to a tool, run it, answer, and check the answer is grounded."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from otp.chat import answers, llm
from otp.chat.guard import ungrounded_numbers
from otp.chat.router import ToolCall, route
from otp.chat.tools import TOOL_SPECS, DataTools

NO_TOOL = {"help", "out_of_scope", "unclear"}
ALLOWED_TOOLS = {t["name"] for t in TOOL_SPECS}  # the LLM can only call these


@dataclass
class Answer:
    text: str
    tool: str
    args: dict = field(default_factory=dict)
    data: dict | None = None
    mode: str = "offline"           # offline | llm | llm-fallback
    grounded: bool = True
    latency_ms: float = 0.0


class Chatbot:
    def __init__(self, db_path: Path, cleaning: dict | None = None, llm_client=None):
        self.tools = DataTools(db_path, cleaning)
        self.llm = llm_client
        self.last: ToolCall | None = None
        self.history: list[dict] = []

    def _run(self, call: ToolCall):
        if call.tool in NO_TOOL:
            return None
        if call.tool not in ALLOWED_TOOLS:
            raise ValueError(f"tool not allowed: {call.tool}")
        return getattr(self.tools, call.tool)(**call.args)

    def ask(self, question: str) -> Answer:
        start = time.perf_counter()
        ans = self._ask_llm(question) if self.llm else None
        if ans is None:
            call = route(question, self.last)
            data = self._run(call)
            ans = Answer(answers.format_answer(call.tool, data or {}), call.tool, call.args, data)
            if call.tool not in NO_TOOL:
                self.last = call
        ans.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return ans

    def _ask_llm(self, question: str) -> Answer | None:
        try:
            tool_use, messages, first = llm.choose_tool(self.llm, question, self.history[-6:])
            if tool_use is None:
                text = " ".join(b.get("text", "") for b in first.get("content", []) if b.get("type") == "text").strip()
                if ungrounded_numbers(text, {}):  # numbers with no tool call = made up
                    text = answers.OUT_OF_SCOPE
                return Answer(text or answers.OUT_OF_SCOPE, "out_of_scope", mode="llm")
            call = ToolCall(tool_use["name"], dict(tool_use.get("input") or {}))
            data = self._run(call)
            text = llm.write_answer(self.llm, messages, first, tool_use, data)
        except Exception:  # API down, bad tool args, etc. -> offline mode
            return None
        bad = ungrounded_numbers(text, data)
        self.last = call
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
        if bad:  # hallucinated numbers: show the safe template answer instead
            return Answer(answers.format_answer(call.tool, data), call.tool, call.args, data,
                          mode="llm-fallback", grounded=False)
        return Answer(text, call.tool, call.args, data, mode="llm")
