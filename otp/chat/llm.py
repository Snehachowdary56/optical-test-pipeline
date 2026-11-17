"""Optional LLM mode: Claude API with tool calling (plain HTTPS, no SDK needed).

Set ANTHROPIC_API_KEY to use it. OTP_LLM_MODEL picks the model.
The model chooses a tool and its arguments; the bot runs the tool, then the model writes the answer
from the tool result only. The groundedness guard checks the answer before it is shown.
"""

from __future__ import annotations

import json
import os
import urllib.request

from otp.chat.tools import TOOL_SPECS

API_URL = "https://api.anthropic.com/v1/messages"
SYSTEM = ("You answer questions about optical component test data for engineers. Always call exactly one tool "
          "to get data; never guess numbers. Answer in 1-3 plain sentences using only numbers from the tool "
          "result. If the question is not about the test data, do not call a tool and say you can only help "
          "with the test data.")


class ClaudeClient:
    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 30):
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]
        self.model = model or os.environ.get("OTP_LLM_MODEL", "claude-haiku-4-5")
        self.timeout = timeout

    def create(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": self.model, "max_tokens": 400, "system": SYSTEM, "messages": messages}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), method="POST", headers={
            "x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)


def choose_tool(client, question: str, history: list[dict]) -> tuple[dict | None, list[dict], dict]:
    """Returns (tool_use block or None, messages so far, raw response)."""
    messages = history + [{"role": "user", "content": question}]
    resp = client.create(messages, TOOL_SPECS)
    tool_use = next((b for b in resp.get("content", []) if b.get("type") == "tool_use"), None)
    return tool_use, messages, resp


def write_answer(client, messages: list[dict], first_response: dict, tool_use: dict, result) -> str:
    messages = messages + [
        {"role": "assistant", "content": first_response["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_use["id"],
                                      "content": json.dumps(result, default=str)}]},
    ]
    resp = client.create(messages, TOOL_SPECS)
    return " ".join(b["text"] for b in resp.get("content", []) if b.get("type") == "text").strip()
