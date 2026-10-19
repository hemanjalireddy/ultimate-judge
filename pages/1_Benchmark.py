"""Benchmark dashboard; the original interactive app remains the main page."""
import json
import os
from pathlib import Path

import plotly.express as px
import streamlit as st

from benchmark import conditions, digest, execute, generate_candidates, write_json
from benchmark_models import Benchmark, BenchmarkRun, Config, Review
from metrics import disagreements, generator_matrix, resource_usage, summary
from models import Bias
from runner import MODELS
from questions import Scenarios, load_catalog, make_scenarios, parse_questions

st.set_page_config(page_title="Ultimate Judge · Benchmark", layout="wide")
st.title("Judge reliability benchmark")
st.caption("Upload questions → choose answer models → freeze answers → judge → inspect results")

with st.expander("1. Upload questions and attach movie facts", expanded=True):
    st.caption("TXT: one question per line. JSON: a list of questions with optional movie_ids and criteria. Facts come from the TMDB CSV, not from the judge.")
    st.download_button("Download sample questions", Path("examples/questions.json").read_bytes(), "questions.json", "application/json")
    uploaded_questions = st.file_uploader("Question file", type=["txt", "json"])
    catalog_upload = st.file_uploader("Movie facts: tmdb_5000_movies.csv (optional if already saved locally)", type="csv")
    catalog_path = Path("data/tmdb_5000_movies.csv")
    catalog_raw = catalog_upload.getvalue() if catalog_upload else (catalog_path.read_bytes() if catalog_path.exists() else None)
    if uploaded_questions and catalog_raw:
        try:
            questions = parse_questions(uploaded_questions.getvalue(), uploaded_questions.name)
            catalog = load_catalog(catalog_raw)
            st.dataframe([q.model_dump() for q in questions], hide_index=True)
            shared_ids = st.multiselect("Shared movie options for questions without movie_ids (choose 2–10)",
                list(catalog), format_func=lambda i: f"{catalog[i].title} · ID {i} · {catalog[i].runtime:g} min")
            if shared_ids:
                st.dataframe([catalog[i].model_dump() for i in shared_ids], hide_index=True)
            if st.button("Save question dataset"):
                scenarios = make_scenarios(questions, catalog_raw, shared_ids)
                destination = Path("data/questions") / f"{digest(scenarios.model_dump(mode='json'))}.json"
                write_json(destination, scenarios)
                st.success(f"Saved {len(scenarios.cases)} questions with movie facts. Continue to answer generation below.")
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Check your upload: {exc}")
    elif uploaded_questions:
        st.info("Upload the movie facts CSV to resolve the questions' movie options.")

with st.expander("2. Choose answer models and build the benchmark", expanded=True):
    scenario_paths = sorted(Path("data/questions").glob("*.json"))
    if Path("data/scenarios.json").exists():
        scenario_paths.append(Path("data/scenarios.json"))
    scenario_source = st.selectbox("Saved question dataset", scenario_paths, format_func=str)
    answer_models = st.multiselect("Answer models (choose exactly two)", MODELS, default=MODELS)
    if scenario_source:
        scenarios = Scenarios.model_validate_json(scenario_source.read_text(encoding="utf-8"))
        st.write(f"{len(scenarios.cases)} questions × 2 models = {len(scenarios.cases) * 2} saved answers.")
        with st.expander("Inspect questions and their facts"):
            st.json(scenarios.model_dump(mode="json"))
    if not os.getenv("GROQ_API_KEY"):
        st.info("Set GROQ_API_KEY in the environment to generate actual AI answers.")
    if st.button("Generate / resume AI answers", disabled=not scenario_source or len(answer_models) != 2 or not os.getenv("GROQ_API_KEY")):
        destination = Path("data/benchmarks") / f"{digest({'scenarios': scenarios.model_dump(mode='json'), 'models': answer_models})}.json"
        try:
            with st.spinner("Generating answers; completed responses are checkpointed…"):
                generate_candidates(scenario_source, destination, answer_models)
            st.success("Benchmark frozen. Select the saved benchmark below to judge these answers.")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: generation stopped. Saved answers are retained; resume with the same questions and models.")

with st.expander("3. Choose judges and run the frozen benchmark", expanded=True):
    sources = [p for p in [Path("examples/demo-benchmark.json"),
                          Path("data/generated-benchmark.json"), Path("data/reviewed-benchmark.json")] if p.exists()]
    sources += [p for p in sorted(Path("data/benchmarks").glob("*.json")) if not p.name.endswith(".progress.json")]
    source = st.selectbox("Frozen benchmark", sources, format_func=str)
    benchmark = Benchmark.model_validate_json(source.read_text(encoding="utf-8")) if source else None
    if benchmark:
        st.caption("Answer models: " + ", ".join(dict.fromkeys(c.generator for case in benchmark.cases for c in case.candidates)))
        st.download_button("Download frozen benchmark", benchmark.model_dump_json(indent=2), "benchmark.json", "application/json")
    mock = st.checkbox("Offline mock judge (no API calls)", value=True)
    judges = st.multiselect("Judges", MODELS, default=MODELS)
    biases = st.multiselect("Experiments", list(Bias), default=list(Bias))
    selected_cases = st.multiselect("Cases (empty means all)", [c.id for c in benchmark.cases] if benchmark else [])
    trials = st.number_input("Trials per order", 1, 100, 3)
    budget = st.number_input("Request budget (including retries; mock mode counts fake calls)", 0, 100000, 2000)
    if st.button("Run / resume", disabled=not benchmark or not judges or not biases):
        config = Config(benchmark=str(source), judges=judges, biases=biases, case_ids=selected_cases,
                        trials=trials, mock=mock, request_budget=budget)
        progress = st.empty()
        waiting = st.empty()
        try:
            run = execute(config, progress=lambda r: progress.write(f"Saved {len(r.observations)} observations"), notice=waiting.info)
            st.success(f"Run saved: {run.id}. Inspect status counts below; a saved run may be partial.")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: execution stopped. Check environment key, inputs or run lock. Saved observations are retained.")

with st.expander("Review a controlled transformation"):
    st.caption("Approval is a human assertion, not automatic semantic validation. Review every edited answer against its original and facts.")
    if benchmark:
        case_id = st.selectbox("Case to review", [c.id for c in benchmark.cases])
        case = next(c for c in benchmark.cases if c.id == case_id)
        index = st.selectbox("Transformation", range(len(case.transformations)),
                             format_func=lambda i: f"{case.transformations[i].bias}: {case.transformations[i].candidate_id}")
        if index is not None:
            transform = case.transformations[index]
            st.write(case.prompt)
            st.json([m.model_dump() for m in case.movie_facts], expanded=False)
            st.text(case.candidates[0 if transform.candidate_id == "c0" else 1].text)
            st.text(transform.text)
            reviewer = st.text_input("Reviewer name")
            assertion = st.text_area("Explain why facts and useful information are unchanged")
            approved = st.checkbox("I have checked this transformation")
            if st.button("Save reviewed snapshot", disabled=not (reviewer.strip() and assertion.strip() and approved)):
                destination = Path("data/reviewed-benchmark.json")
                if destination.exists() and source != destination:
                    st.error("Select the existing reviewed snapshot above before adding another review; this preserves prior annotations.")
                else:
                    transform.review, transform.reviewer, transform.assertion = Review.APPROVED, reviewer, assertion
                    write_json(destination, benchmark)
                    st.success("Saved data/reviewed-benchmark.json. Select it above to retain prior reviews and run the reviewed inputs.")

paths = sorted(Path("results/benchmarks").glob("*.json"))
paths += sorted(Path("examples/results").glob("*.json"))
if not paths:
    st.info("Run the offline demo to populate the explorer.")
    st.stop()
path = st.selectbox("Saved run", paths, format_func=str)
run = BenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))
if run.config.mock:
    st.warning("MOCK RESULTS — deterministic software demonstration. These are not findings about Groq models.")
model_filter = st.multiselect("Filter judges", run.config.judges, default=run.config.judges)
case_filter = st.multiselect("Filter scenarios", [c.id for c in run.benchmark.cases])
bias_filter = st.multiselect("Filter conditions", [Bias.POSITION, Bias.VERBOSITY, Bias.STYLE],
                             default=[Bias.POSITION, Bias.VERBOSITY, Bias.STYLE])
trial_filter = st.multiselect("Filter trial numbers", range(run.config.trials), default=list(range(run.config.trials)))
filtered = run.model_copy(deep=True)
filtered.config.judges = model_filter
filtered.observations = [o for o in run.observations if o.judge in model_filter and o.bias in bias_filter
                        and o.trial in trial_filter and (not case_filter or o.case_id in case_filter)]
overview, explorer, analysis, comparison = st.tabs(["Overview", "Experiment Explorer", "Bias Analysis", "Judge Comparison"])
rows = summary(filtered)
with overview:
    st.metric("Recorded observations", len(filtered.observations))
    st.caption(f"Run {run.id} · snapshot {run.benchmark.snapshot} · seed {run.config.seed}")
    st.dataframe([{**r, "ci95": str(r["ci95"])} for r in rows], hide_index=True)
    st.dataframe(resource_usage(filtered), hide_index=True)
    st.caption("n is the usable comparison denominator, not the number of independent cases. Missing values mean no eligible data. API failures are excluded from preference metrics.")
    st.download_button("Download complete run", run.model_dump_json(indent=2), f"{run.id}.json", "application/json")
with explorer:
    if filtered.observations:
        i = st.selectbox("Judgment", range(len(filtered.observations)), format_func=lambda i:
                         f"{filtered.observations[i].case_id} · {filtered.observations[i].judge} · {filtered.observations[i].bias} · trial {filtered.observations[i].trial} · {filtered.observations[i].order}")
        observation = filtered.observations[i]
        case = next(c for c in run.benchmark.cases if c.id == observation.case_id)
        texts = next(texts for bias, target, texts in conditions(case, run.config)
                     if bias == observation.bias and target == observation.target)
        st.write(case.prompt)
        st.dataframe([m.model_dump() for m in case.movie_facts], hide_index=True)
        for column, label, cid in zip(st.columns(2), ("Displayed A", "Displayed B"), observation.order):
            column.subheader(label)
            column.write(texts[0 if cid == "c0" else 1])
        st.write(f"Status: {observation.status}; canonical winner: {observation.winner}")
        if observation.decision:
            st.write(observation.decision.reason)
        st.json(observation.model_dump(mode="json"), expanded=False)
with analysis:
    chart = [{"Judge": r["judge"], "Metric": r["metric"], "Rate": r["value"],
              "n": r["n"], "Cases": r["cases"],
              "upper": max(0, r["ci95"][1] - r["value"]) if r["ci95"] else 0,
              "lower": max(0, r["value"] - r["ci95"][0]) if r["ci95"] else 0}
             for r in rows if r["value"] is not None]
    if chart:
        st.plotly_chart(px.bar(chart, x="Metric", y="Rate", color="Judge", barmode="group",
                              error_y="upper", error_y_minus="lower", hover_data=["n", "Cases"], range_y=[0, 1]))
    st.caption("95% percentile bootstrap intervals resample whole cases (seed from run). No interval is shown with fewer than two cases. Reviewed style/verbosity metrics exclude unreviewed transformations. Preference is not proof of unfair bias.")
with comparison:
    st.dataframe([{**r,"ci95":str(r["ci95"])} for r in generator_matrix(filtered)], hide_index=True)
    st.dataframe(disagreements(filtered), hide_index=True)
    st.caption("Disagreements match case, condition, target, trial and order. Self-preference uses only decisive baseline judgments where exactly one answer came from the judge. Template candidates are ineligible.")
