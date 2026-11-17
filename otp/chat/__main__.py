"""Chat in the terminal:  python -m otp.chat   (add --llm to use the Claude API if ANTHROPIC_API_KEY is set)."""

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from otp import generate, pipeline
from otp.chat import Chatbot


def build_bot(raw_dir: Path = Path("data/raw"), output_dir: Path = Path("output"), use_llm: bool = False) -> Chatbot:
    if not any(raw_dir.glob("*.csv")):
        generate.write_raw_csvs(generate.generate_measurements(), raw_dir)
    result = pipeline.run(raw_dir, output_dir)
    client = None
    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        from otp.chat.llm import ClaudeClient
        client = ClaudeClient()
    return Chatbot(result.db_path, asdict(result.cleaning), client)


def main() -> None:
    p = argparse.ArgumentParser(description="Optical test data chatbot")
    p.add_argument("--llm", action="store_true", help="use the Claude API (needs ANTHROPIC_API_KEY)")
    a = p.parse_args()
    bot = build_bot(use_llm=a.llm)
    print("Optical test data chatbot. Ask a question, or type 'quit'.  (mode: %s)" % ("llm" if bot.llm else "offline"))
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print("Bot:", bot.ask(q).text)


if __name__ == "__main__":
    main()
