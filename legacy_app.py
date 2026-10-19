"""Run with: streamlit run app.py"""
import csv
import io
import json
import os

import streamlit as st
from groq import Groq

from models import Bias, Case, Movie, Run, Variant
from runner import MODELS, RESULTS, RUBRIC, evaluate, generate, save, summarize
from provider import make_client

st.set_page_config(page_title="Ultimate Judge", layout="wide")
st.title("Ultimate Judge")
st.page_link("pages/1_Benchmark.py", label="Upload questions → generate answers → judge results", icon="📋")
st.caption("Compare movie answers, then inspect how judge decisions change.")
key = bool(os.getenv("GROQ_API_KEY"))
st.sidebar.caption("Groq key configured in environment" if key else "Set GROQ_API_KEY in your environment for live calls.")
st.sidebar.markdown("[Get a Groq key](https://console.groq.com/keys)")
answer_models = st.sidebar.multiselect("Answer models (choose two)", MODELS, default=MODELS)
judge_models = st.sidebar.multiselect("Judge models", MODELS, default=MODELS)
biases = st.sidebar.multiselect("Bias experiments", list(Bias), default=list(Bias))

setup, results = st.tabs(["Experiment", "Saved results"])
with setup:
    st.markdown("Upload **tmdb_5000_movies.csv** from the [TMDB dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).")
    upload = st.file_uploader("Movie dataset", type="csv")
    if upload:
        try:
            movies = []
            for row in csv.DictReader(io.StringIO(upload.getvalue().decode("utf-8-sig"))):
                if row.get("runtime") and float(row["runtime"]) > 0 and row.get("overview"):
                    movies.append(Movie(id=int(row["id"]), title=row["title"],
                        genres=[g["name"] for g in json.loads(row["genres"])],
                        runtime=float(row["runtime"]), overview=row["overview"]))
            selected = st.multiselect("Movie options (pick 2–5)", range(len(movies)),
                format_func=lambda i: f"{movies[i].title} ({movies[i].runtime:g} min)")
            request = st.text_area("Viewer request", "Recommend one science-fiction movie under 120 minutes. Explain why it fits.")
            if selected:
                st.dataframe([movies[i].model_dump() for i in selected], hide_index=True)
            if st.button("Generate answers", disabled=not key or len(answer_models) != 2 or not 2 <= len(selected) <= 5 or not request.strip()):
                case = Case(request=request, movies=[movies[i] for i in selected])
                with st.spinner("Generating answers…"):
                    client = make_client()
                    answers = tuple(generate(client, model, case) for model in answer_models)
                st.session_state.draft = (case, answers)
        except Exception as error:
            st.error(f"Could not prepare answers: {type(error).__name__}. Check the CSV, API key, and model access.")

    if "draft" in st.session_state:
        case, answers = st.session_state.draft
        st.subheader("Review answers and controlled variants")
        st.write(case.request)
        for i, answer in enumerate(answers):
            with st.expander(f"Answer {i + 1} — {answer.model}", expanded=True):
                st.write(answer.text)
                if Bias.VERBOSITY in biases:
                    answer.verbose = st.text_area("Longer version: repeat or elaborate without adding claims", value=answer.text,
                        key=f"verbose-{i}-{hash(answer.text)}")
                if Bias.STYLE in biases:
                    answer.styled = st.text_area("Style version: change formatting only; retain the wording", value=answer.text,
                        key=f"styled-{i}-{hash(answer.text)}")
        reviewed = st.checkbox("I reviewed the variants: the useful information is unchanged.")
        valid = bool(judge_models and biases)
        if Bias.SELF_PREFERENCE in biases and not set(judge_models).intersection(a.model for a in answers):
            st.warning("Self-preference needs a judge that also generated an answer.")
            valid = False
        for answer in answers:
            if Bias.VERBOSITY in biases and len(answer.verbose.split()) <= len(answer.text.split()):
                valid = False
            if Bias.STYLE in biases and (not answer.styled.strip() or answer.styled == answer.text):
                valid = False
        if not valid:
            st.info("Select judges/biases and provide the selected variants for both answers. Verbosity variants must be longer.")
        if st.button("Create experiment", disabled=not valid or not reviewed):
            run = Run(case=case, answers=answers, judges=judge_models, biases=biases, rubric=RUBRIC).model_copy(deep=True)
            save(run)
            st.session_state.active = run
        if "active" in st.session_state:
            run = st.session_state.active
            st.caption(f"Active experiment: {run.id}. Settings are frozen when created.")
            if st.button("Run / resume judging", disabled=not key):
                progress = st.empty()
                table = st.empty()
                try:
                    client = make_client()
                    for judgment in evaluate(client, run):
                        progress.write(f"Saved {len(run.judgments)} judgments — {judgment.judge}")
                        table.dataframe(summarize(run), hide_index=True)
                    st.success("Evaluation complete.")
                except Exception as error:
                    st.error(f"Stopped: {type(error).__name__}. Completed judgments are saved. Check your key, model access or rate limits, then resume.")

with results:
    st.caption("Metrics are exploratory. Own-answer preference is not proof of bias; quality can explain it. Small samples and ordinary model variability can also change verdicts.")
    imported = st.file_uploader("Import a contributor's experiment", type="json")
    if imported and st.button("Save imported experiment"):
        try:
            run = Run.model_validate_json(imported.getvalue())
            if run.version != "1" or any(j.order not in [(0, 1), (1, 0)] for j in run.judgments):
                raise ValueError("Unsupported version or answer order")
            # Avoid replacing an existing local run from another contributor.
            from uuid import uuid4
            run.id = uuid4().hex
            save(run)
            st.success("Imported.")
        except ValueError as error:
            st.error(f"Invalid experiment: {error}")

    @st.fragment(run_every="5s")
    def show_results():
        files = sorted(RESULTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            st.info("Saved experiments will appear here.")
            return
        file = st.selectbox("Experiment", files, format_func=lambda p: p.stem)
        try:
            run = Run.model_validate_json(file.read_text(encoding="utf-8"))
        except ValueError:
            st.error("This saved file is not a valid experiment.")
            return
        st.write(run.case.request)
        st.caption(f"{run.created.isoformat()} · {len(run.judgments)} completed judgments")
        judges = st.multiselect("Filter judges", run.judges, default=run.judges, key=f"filter-{run.id}")
        st.dataframe([row for row in summarize(run) if row["judge"] in judges], hide_index=True)
        st.caption("Win-rate changes compare modified answers to their originals in the same position. Ties count as non-wins. Position compares answer identities after swapping.")
        st.dataframe([{"judge": j.judge, "variant": j.variant.value,
            "modified answer": j.target + 1 if j.variant != Variant.ORIGINAL else None,
            "A from": run.answers[j.order[0]].model, "B from": run.answers[j.order[1]].model,
            "verdict": j.decision.verdict.value, "reason": j.decision.reason}
            for j in run.judgments if j.judge in judges], hide_index=True)
        with st.expander("Inputs and answers"):
            st.json({"case": run.case.model_dump(), "answers": [a.model_dump() for a in run.answers], "rubric": run.rubric})
        st.download_button("Download experiment JSON", run.model_dump_json(indent=2), f"{file.stem}.json", "application/json")
        if st.button("Load for resume", key=f"resume-{run.id}"):
            st.session_state.active = run
            draft = run.model_copy(deep=True)
            st.session_state.draft = (draft.case, draft.answers)
            st.rerun()

    show_results()
