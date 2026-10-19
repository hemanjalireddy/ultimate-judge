"""Primary flow: questions -> model answers -> judges -> results."""
import os
from pathlib import Path

import streamlit as st

from benchmark import digest, execute, generate_candidates, write_json
from benchmark_models import Benchmark, BenchmarkRun, Config, Review
from metrics import summary
from models import Bias
from provider import ProviderError, Status
from questions import make_scenarios, parse_questions
from runner import MODELS

st.set_page_config(page_title="Ultimate Judge", layout="wide")
st.title("Ultimate Judge")
st.caption("Upload questions. Compare two models' answers. Run the judges.")
if Path("pages/2_Dashboard.py").exists():
    st.page_link("pages/2_Dashboard.py", label="Open performance dashboard", icon="📊")

st.subheader("1. Upload your questions")
upload = st.file_uploader("Questions file — TXT or JSON", type=["txt", "json"])
sample_path = Path(__file__).resolve().parent / "examples" / "30_questions.txt"
try:
    sample_questions = sample_path.read_bytes()
except OSError:
    st.caption("Example download is unavailable. You can still upload your own question file.")
else:
    st.download_button("Download 30 example questions", sample_questions,
                       "30_questions.txt", "text/plain")
answer_models = st.multiselect("Answer models — choose two", MODELS, default=MODELS)
st.caption("Movie facts are loaded automatically from the local catalog. Both models receive the same facts.")
scenarios = None
if upload:
    try:
        catalog = Path("data/tmdb_5000_movies.csv")
        if not catalog.exists():
            st.error("The server's movie catalog is missing. Restore data/tmdb_5000_movies.csv to enable this app.")
        else:
            questions = parse_questions(upload.getvalue(), upload.name)
            scenarios = make_scenarios(questions, catalog.read_bytes())
            st.success(f"{len(questions)} questions ready.")
            with st.expander("Preview questions and movie facts"):
                st.caption("Each question uses up to six options selected by genre/title keywords. This is not a full constraint search. JSON movie_ids override the selection.")
                for case in scenarios.cases:
                    st.write(case.prompt)
                    st.dataframe([m.model_dump() for m in case.movie_facts], hide_index=True)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"Please check the question file: {exc}")

# Changing input/model selection hides the previous answers and verdicts immediately.
input_id = digest({"scenarios": scenarios.model_dump(mode="json"), "models": answer_models}) if scenarios and len(answer_models) == 2 else None
if st.session_state.get("question_input_id") != input_id:
    st.session_state.question_input_id = input_id
    st.session_state.pop("question_benchmark", None)
    st.session_state.pop("question_run", None)

if not os.getenv("GROQ_API_KEY"):
    st.info("Live calls need GROQ_API_KEY in the server environment (.env when using Docker).")
if st.button("Run test", type="primary", disabled=input_id is None):
    if not os.getenv("GROQ_API_KEY"):
        st.error("Set GROQ_API_KEY, then restart the app and click Run test.")
    else:
        question_path = Path("data/questions") / f"{input_id}.json"
        frozen_path = Path("data/benchmarks") / f"{input_id}.json"
        try:
            write_json(question_path, scenarios)
            with st.spinner("Asking both models… Completed answers are saved so you can resume."):
                generate_candidates(question_path, frozen_path, answer_models)
            st.session_state.question_benchmark = str(frozen_path)
        except (ProviderError, ValueError, OSError) as exc:
            st.error(f"Generation stopped: {exc}. Click Run test again to resume saved progress.")

if "question_benchmark" in st.session_state:
    frozen_path = Path(st.session_state.question_benchmark)
    benchmark = Benchmark.model_validate_json(frozen_path.read_text(encoding="utf-8"))
    st.subheader("2. Compare the answers")
    st.caption("These model responses are saved. Judging reuses this exact text.")
    case_index = st.selectbox("Question", range(len(benchmark.cases)),
        format_func=lambda i: benchmark.cases[i].prompt, key=f"question-{input_id}")
    case = benchmark.cases[case_index]
    st.write(case.prompt)
    for column, answer in zip(st.columns(2), case.candidates):
        column.subheader(answer.generator)
        column.write(answer.text)
    st.download_button("Download benchmark with answers", benchmark.model_dump_json(indent=2),
                       "benchmark.json", "application/json")

    st.subheader("3. Run the judges")
    judges = st.multiselect("Judge models", MODELS, default=MODELS)
    biases = st.multiselect("Experiments", list(Bias), default=[Bias.POSITION, Bias.SELF_PREFERENCE, Bias.VERBOSITY])
    st.caption("Position and self-preference reuse the original pair. Verbosity and style use variants of each answer.")
    if Bias.VERBOSITY in biases or Bias.STYLE in biases:
        st.info("Unreviewed variants can be inspected, but are excluded from controlled preference metrics. Use the Benchmark page to review transformations.")
    with st.expander("Run settings"):
        trials = st.number_input("Trials per order", 1, 100, 3)
        budget = st.number_input("Request budget, including retries", 0, 100000, 2000)
    planned = len(benchmark.cases) * len(judges) * trials * 2 * (1 + 2 * (Bias.VERBOSITY in biases) + 2 * (Bias.STYLE in biases))
    st.caption(f"Judging covers all {len(benchmark.cases)} questions: {planned} judgments before retries.")
    config = Config(benchmark=str(frozen_path), judges=judges, biases=biases, trials=trials,
                    request_budget=budget, mock=False) if judges and biases else None
    config_id = digest(config.model_dump(mode="json")) if config else None
    if st.session_state.get("question_judge_config") != config_id:
        st.session_state.question_judge_config = config_id
        st.session_state.pop("question_run", None)
    if st.button("Run judge", type="primary", disabled=config is None):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Set GROQ_API_KEY to run judges.")
        else:
            progress = st.empty()
            waiting = st.empty()
            try:
                run = execute(config, progress=lambda r: progress.write(f"Completed {sum(o.status != Status.API_ERROR for o in r.observations)} of {planned} comparisons"), notice=waiting.info)
                if all(o.status != Status.API_ERROR for o in run.observations):
                    waiting.empty()
                st.session_state.question_run = run.model_dump_json()
            except (ProviderError, ValueError, OSError) as exc:
                st.error(f"Judging stopped: {exc}. Saved results remain available; click Run judge to resume.")

    if "question_run" in st.session_state:
        run = BenchmarkRun.model_validate_json(st.session_state.question_run)
        st.subheader("4. Results")
        st.write(f"{sum(o.status != Status.API_ERROR for o in run.observations)} of {planned} comparisons completed. API failures will be retried on resume.")
        if run.failed_attempts:
            st.caption(f"{len(run.failed_attempts)} earlier failed attempts retained in the exported history.")
        st.dataframe([{"judge": o.judge, "condition": o.bias.value, "trial": o.trial + 1,
                       "status": o.status.value, "winner": o.winner,
                       "reason": o.decision.reason if o.decision else o.error}
                      for o in run.observations if o.case_id == case.id], hide_index=True)
        st.caption("c0/c1 identify the saved answer pair for the original comparison. In style/verbosity comparisons, c0 is the original and c1 its variant.")
        st.dataframe([{**row, "ci95": str(row["ci95"])} for row in summary(run)], hide_index=True)
        st.download_button("Download judge results", run.model_dump_json(indent=2), "results.json", "application/json")

st.page_link("pages/1_Benchmark.py", label="Saved runs, transformation review and detailed analysis")
