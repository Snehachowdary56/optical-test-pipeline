# Optical Component Test Data Pipeline

A Python data pipeline that turns raw test measurements from optical components into clean data, validation statistics and an HTML report for engineering review, plus an **AI chatbot** that lets engineers ask questions about the results in plain English.

![CI](https://github.com/Snehachowdary56/optical-test-pipeline/actions/workflows/ci.yml/badge.svg)

## What it does

```
raw CSVs ──► load ──► clean ──► transform ──► DuckDB (SQL) ──► statistics ──► HTML report
```

1. **Load**: reads every CSV in a folder (one per test station) and checks the required columns are present.
2. **Clean**: removes duplicate rows, rows with missing values, and physically impossible readings such as `-999` sentinel values from the instrument. It reports how many rows were dropped and why.
3. **Transform**: converts optical power from dBm to mW, flags each reading against the spec limits, and summarizes results per device and per temperature.
4. **Store**: saves the clean measurements and device summaries as SQL tables in a DuckDB database, then queries them with SQL.
5. **Analyze**: computes descriptive statistics (mean, standard deviation, percentiles), yield (the share of devices that pass every test), and a linear regression of insertion loss against temperature, with R².
6. **Report**: writes a self-contained HTML report with key metrics, tables and charts.

Because there's no real lab data, `otp/generate.py` creates realistic sample data: 200 devices measured at 3 temperatures and 8 C-band wavelengths, with a built-in temperature effect and deliberately injected problems (missing values, duplicates, instrument glitches) for the pipeline to clean.

## Tech stack

Python · Pandas · NumPy · DuckDB (SQL) · Matplotlib · Streamlit · Claude API (optional) · pytest · GitHub Actions

## Run it

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m otp --generate
```

Then open `output/report.html` in your browser.

Example output:

```
Generated 4848 raw rows into 3 CSV files in data/raw/
Cleaned: 4848 rows in -> 4728 rows out (48 duplicates, 48 missing, 24 out of range)
Devices: 200 | Yield: 98.0%
Loss vs temperature: 0.00798 dB/°C (R² = 0.478)
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--generate` | off | Create sample data first |
| `--devices` | 200 | Number of devices to generate |
| `--raw-dir` | `data/raw` | Folder of input CSVs |
| `--output-dir` | `output` | Where the report and database go |

### Run the tests

```bash
pytest
```

42 tests: 18 cover cleaning, unit conversion, spec checks, statistics and a full end-to-end run; 24 cover the chatbot. The end-to-end test checks that the pipeline recovers the temperature coefficient that was built into the generated data.

## AI chatbot: ask the test data questions

Engineers often just want a quick answer ("Which devices failed?", "Is loss worse at 75°C?") without opening the report or writing SQL. The chatbot answers in plain English and remembers context for follow-ups.

```
You: What is the yield?
Bot: Yield is 98.0%: 196 of 200 devices passed every test and 4 failed.
You: what about station B?
Bot: Yield at STN-B is 95.5%: 64 of 67 devices passed every test and 3 failed.
You: How does temperature affect loss?
Bot: Insertion loss rises about 0.00798 dB per °C (linear regression, R² = 0.478). Mean loss by temperature: -5°C: 1.774 dB, 25°C: 2.015 dB, 75°C: 2.412 dB.
You: What is the capital of France?
Bot: Sorry, I can only answer questions about the optical component test data. ...
```

### How it works (AI application design)

```
question ──► pick a tool ──► run SQL on DuckDB ──► write the answer ──► groundedness check ──► answer
             (rules or LLM)   (11 tools)            (template or LLM)    (numbers must come from the data)
```

- **Tools, not guesses.** Every number comes from one of 11 SQL tools (yield, stations, statistics, devices, temperature regression, cleaning, spec limits). The model never computes numbers itself.
- **Two modes.** *Offline* (default): a rule-based router picks the tool, so it is free, fast and reproducible. *LLM*: with `ANTHROPIC_API_KEY` set, a large language model (Claude API, tool calling) picks the tool and writes the answer, which handles phrasings the rules miss.
- **Guardrails for reliability.**
  - The LLM can only call the 11 listed tools (allow-list), never raw SQL.
  - The **groundedness guard** checks that every number in an LLM answer appears in the tool result. If not (a hallucination), the bot shows the safe template answer instead.
  - Off-topic questions are refused, and if the API fails the bot falls back to offline mode.
  - Device IDs are passed as SQL parameters, so a question can't inject SQL.

### Evaluation (accuracy, reliability, usability)

`python -m otp.chat.evaluate` runs 60 labelled questions (45 single questions, 5 off-topic, 5 two-turn follow-ups). The expected answers are computed **independently with pandas**, while the bot uses SQL, so a wrong query or wrong routing shows up as a miss. Results are in [`eval/results.md`](eval/results.md), and CI fails if answer accuracy drops below 95%.

| Metric (offline mode) | Result |
|---|---|
| Answer accuracy | 100% (60 questions) |
| Right tool chosen | 100% |
| Off-topic refusals | 100% |
| Follow-up questions | 100% |
| Groundedness (no numbers missing from the data) | 100% |
| Median latency | about 20 ms |

What the evaluation found: the first run scored 90.9% groundedness. Five template answers contained numbers that weren't in the tool data (a hard-coded "-999" and a count calculated in the template). Both were fixed so every number now comes from the data.

Honest limits: the test questions were written by the developer, so new phrasings will score lower in offline mode, and that's where the LLM mode helps. The LLM mode is unit-tested with a fake model (good answer, hallucinated numbers, off-topic, API error, attempt to call an internal function); it has not been scored on the question set, because that needs an API key.

### Run it

```bash
python -m otp.chat                 # chat in the terminal (offline)
streamlit run app_chat.py          # web chat in your browser
python -m otp.chat.evaluate        # score it
```

Optional LLM mode (Windows Command Prompt): `set ANTHROPIC_API_KEY=your-key` then `python -m otp.chat --llm`.

## Project structure

```
otp/
  generate.py   Sample data with realistic noise and injected problems
  load.py       Reads and validates raw CSVs
  clean.py      Removes duplicates, missing values and impossible readings
  transform.py  Unit conversion, spec flags, per-device and per-temperature summaries
  storage.py    DuckDB tables and SQL queries
  stats.py      Descriptive statistics, yield, linear regression
  report.py     HTML report with charts
  pipeline.py   Runs every step in order
  chat/         AI chatbot: tools (SQL), router, LLM client, guard, evaluation
app_chat.py     Streamlit web chat
eval/           Latest chatbot evaluation results
tests/          pytest tests
```

## Design decisions

- **Each step is a separate, testable function**, so cleaning rules or statistics can change without touching the rest.
- **Bad data is counted, not silently dropped.** The report states how many rows were removed and why, so engineers can trust the numbers.
- **Pass/fail is per device, worst case.** A device fails if any single reading is out of spec, which is how validation usually works.
- **DuckDB** is used because it runs SQL analytics directly on DataFrames with no server to set up.

## Ideas for next steps

- Outlier detection with statistical rules (e.g. z-score or IQR per test condition)
- Station-to-station comparison with a hypothesis test (e.g. ANOVA)
- Process capability metrics (Cpk) against the spec limits
- Read larger datasets directly with DuckDB or Polars instead of Pandas
