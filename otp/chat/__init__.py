"""Chatbot that answers engineers' questions about optical component test data.

Every answer is computed with SQL over the pipeline's DuckDB database (tool calls), so the bot
never invents numbers. Two modes:
  * offline (default): a rule-based intent router picks the tool; no API key, fully reproducible.
  * llm: a large language model (Claude API, tool calling) picks the tool and writes the answer;
    a groundedness guard rejects any answer whose numbers are not in the tool results.
"""
from otp.chat.bot import Answer, Chatbot

__all__ = ["Answer", "Chatbot"]
