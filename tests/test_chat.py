from dataclasses import asdict
from pathlib import Path

import pytest

from otp import generate, pipeline
from otp.chat import Chatbot
from otp.chat.guard import ungrounded_numbers
from otp.chat.router import ToolCall, route


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chat")
    generate.write_raw_csvs(generate.generate_measurements(60, seed=7), tmp / "raw")
    return pipeline.run(tmp / "raw", tmp / "out")


@pytest.fixture
def bot(res):
    return Chatbot(res.db_path, asdict(res.cleaning))


# ---- router ---------------------------------------------------------------------------
@pytest.mark.parametrize("q,tool,args", [
    ("What is the yield?", "device_yield", {}),
    ("yield at station b", "device_yield", {"station": "STN-B"}),
    ("Compare stations", "yield_by_station", {}),
    ("average output power at 75C", "metric_stats", {"metric": "output_power_dbm", "temperature_c": 75}),
    ("mean loss at -5 °C", "metric_stats", {"metric": "insertion_loss_db", "temperature_c": -5}),
    ("tell me about dev 7", "device_detail", {"device_id": "DEV-0007"}),
    ("top 3 worst devices", "worst_devices", {"n": 3}),
    ("how many duplicates?", "cleaning_summary", {}),
    ("what is the capital of France?", "out_of_scope", {}),
])
def test_router(q, tool, args):
    call = route(q)
    assert call.tool == tool and call.args == args


def test_router_follow_up_keeps_context():
    prev = ToolCall("metric_stats", {"metric": "insertion_loss_db", "temperature_c": 25})
    assert route("what about at 75C?", prev).args == {"metric": "insertion_loss_db", "temperature_c": 75}


# ---- tools (SQL) agree with pandas ----------------------------------------------------
def test_yield_matches_pandas(bot, res):
    r = bot.tools.device_yield()
    assert r["devices"] == len(res.device_summary)
    assert r["passed"] == int(res.device_summary["passed"].sum())


def test_metric_stats_match_pandas(bot, res):
    m = res.measurements[res.measurements["temperature_c"] == 75]["insertion_loss_db"]
    r = bot.tools.metric_stats("insertion_loss_db", temperature_c=75)
    assert r["n"] == len(m) and r["mean"] == pytest.approx(m.mean(), abs=5e-4)


def test_unknown_device(bot):
    assert "couldn't find" in bot.ask("show DEV-9999").text


def test_sql_injection_is_just_data(bot):
    a = bot.ask("tell me about DEV-0001'; DROP TABLE measurements; --")
    assert a.tool == "device_detail"
    assert bot.tools.overview()["measurements"] > 0


# ---- guard ----------------------------------------------------------------------------
def test_guard_accepts_rounded_and_percent():
    assert ungrounded_numbers("Yield is 98% (196 of 200), loss 2.41 dB", {"y": 0.98, "p": 196, "d": 200, "l": 2.4123}) == []


def test_guard_flags_made_up_number():
    assert ungrounded_numbers("Yield is 97.5%", {"yield_pct": 98.0}) == [97.5]


# ---- LLM mode with a fake model (no API key needed) -----------------------------------
class FakeLLM:
    def __init__(self, tool=None, args=None, answer="", fail=False):
        self.tool, self.args, self.answer, self.fail, self.calls = tool, args or {}, answer, fail, 0

    def create(self, messages, tools=None):
        if self.fail:
            raise ConnectionError("API down")
        self.calls += 1
        if self.calls == 1 and self.tool:
            return {"content": [{"type": "tool_use", "id": "t1", "name": self.tool, "input": self.args}]}
        return {"content": [{"type": "text", "text": self.answer}]}


def test_llm_grounded_answer_is_used(res):
    y = Chatbot(res.db_path).tools.device_yield()
    fake = FakeLLM("device_yield", {}, f"{y['passed']} of {y['devices']} devices passed, a yield of {y['yield_pct']}%.")
    a = Chatbot(res.db_path, llm_client=fake).ask("how are we doing?")
    assert a.mode == "llm" and a.grounded and str(y["passed"]) in a.text


def test_llm_hallucination_falls_back_to_template(res):
    fake = FakeLLM("device_yield", {}, "Yield is 12.3% across 999 devices.")
    a = Chatbot(res.db_path, llm_client=fake).ask("what's the yield?")
    assert a.mode == "llm-fallback" and not a.grounded
    assert "12.3" not in a.text and a.text.startswith("Yield is")


def test_llm_off_topic_with_made_up_number_is_refused(res):
    a = Chatbot(res.db_path, llm_client=FakeLLM(None, answer="Paris has 2.1 million people.")).ask("capital of France?")
    assert "only answer questions" in a.text


def test_llm_error_uses_offline_mode(res):
    a = Chatbot(res.db_path, llm_client=FakeLLM(fail=True)).ask("What is the yield?")
    assert a.mode == "offline" and a.tool == "device_yield"


def test_llm_cannot_call_internal_functions(res):
    fake = FakeLLM("_q", {"sql": "DROP TABLE measurements"})
    a = Chatbot(res.db_path, llm_client=fake).ask("What is the yield?")
    assert a.mode == "offline" and Chatbot(res.db_path).tools.overview()["measurements"] > 0


def test_llm_bad_tool_name_uses_offline_mode(res):
    a = Chatbot(res.db_path, llm_client=FakeLLM("drop_tables", {})).ask("What is the yield?")
    assert a.mode == "offline"


# ---- evaluation runs ------------------------------------------------------------------
def test_evaluation_meets_threshold(tmp_path):
    from otp.chat import evaluate
    s = evaluate.run(min_accuracy=0.95, out_dir=tmp_path)
    assert s["groundedness"] == 1.0 and s["refusal_accuracy"] == 1.0
    assert (tmp_path / "results.md").exists()


def test_streamlit_app_answers(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(tmp_path)
    app = Path(__file__).resolve().parents[1] / "app_chat.py"
    monkeypatch.syspath_prepend(str(app.parent))
    at = AppTest.from_file(str(app), default_timeout=60).run()
    at.chat_input[0].set_value("What is the yield?").run()
    assert any("Yield is" in m.markdown[0].value for m in at.chat_message if m.markdown)
