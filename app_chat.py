"""Web chat for the optical test data chatbot.   Run:  streamlit run app_chat.py"""

import os

import streamlit as st

from otp.chat.__main__ import build_bot

st.set_page_config(page_title="Optical Test Data Chatbot")
st.title("Optical test data chatbot")
st.caption("Ask about yield, stations, insertion loss, output power, temperature, a device (e.g. DEV-0042) or data cleaning. "
           "Answers come from SQL queries on the test database.")

use_llm = st.sidebar.toggle("Use LLM (Claude API)", value=False, disabled=not os.environ.get("ANTHROPIC_API_KEY"),
                            help="Needs ANTHROPIC_API_KEY. Off = offline rule-based mode.")
if "bot" not in st.session_state or st.session_state.get("llm") != use_llm:
    st.session_state.bot = build_bot(use_llm=use_llm)
    st.session_state.llm = use_llm
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            st.caption(m["meta"])

if q := st.chat_input("e.g. What is the yield at station B?"):
    st.session_state.messages.append({"role": "user", "content": q})
    a = st.session_state.bot.ask(q)
    meta = f"tool: {a.tool} {a.args or ''} · mode: {a.mode} · {a.latency_ms} ms"
    st.session_state.messages.append({"role": "assistant", "content": a.text, "meta": meta})
    st.rerun()

for example in ("What is the yield?", "Compare stations", "Average insertion loss at 75C", "Which devices failed?"):
    if st.sidebar.button(example):
        a = st.session_state.bot.ask(example)
        st.session_state.messages += [{"role": "user", "content": example},
                                      {"role": "assistant", "content": a.text, "meta": f"tool: {a.tool} · mode: {a.mode}"}]
        st.rerun()
