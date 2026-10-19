"""Read-only dashboard over one saved benchmark run; never makes model calls."""
from collections import Counter
from pathlib import Path
from statistics import quantiles

import plotly.graph_objects as go
import streamlit as st

from benchmark_models import BenchmarkRun, Review
from metrics import summary
from provider import Status

st.set_page_config(page_title="Ultimate Judge · Dashboard", layout="wide")
st.markdown("""<style>
.stApp {background:#0e1520; color:#e9edf5}
[data-testid="stSidebar"] {background:#121b28}
[data-testid="stVerticalBlockBorderWrapper"] {background:#151f2c; border-radius:12px}
[data-testid="stMetricValue"] {color:#80dbad; font-size:1.9rem}
h1,h2,h3 {color:#f3f5fa}
</style>""", unsafe_allow_html=True)
st.title("Ultimate Judge · Performance Dashboard")
st.caption("Explore judge reliability and preference patterns across your saved movie comparisons.")


def percent(value):
    return "—" if value is None else f"{value:.1%}"


def p95(observations):
    values = [o.completion.latency_seconds for o in observations if o.completion]
    return quantiles(values, n=100, method="inclusive")[94] if len(values) > 1 else (values[0] if values else None)


def chart(figure):
    figure.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", height=310,
                         margin=dict(l=10, r=10, t=20, b=15),
                         colorway=["#80dbad", "#9b87f5", "#59afff", "#f2c66d"],
                         legend=dict(orientation="h", y=1.16),
                         font=dict(color="#dce4ef"))
    st.plotly_chart(figure, width="stretch")


source = st.sidebar.radio("Results source", ["Saved runs", "Offline demo"])
folder = Path("results/benchmarks" if source == "Saved runs" else "examples/results")
paths = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not paths:
    st.info("No saved judge runs yet. Generate answers and run the judges on the main page first.")
    st.stop()
path = st.sidebar.selectbox("Run", paths, format_func=lambda p: p.stem)
auto = st.sidebar.toggle("Refresh every 10 seconds", value=False)
st.sidebar.button("Refresh now")
st.sidebar.caption("Refresh reads saved progress. It does not start or resume API calls.")


@st.cache_data(show_spinner=False, max_entries=8)
def statistics(raw):
    return summary(BenchmarkRun.model_validate_json(raw))


@st.fragment(run_every="10s" if auto else None)
def dashboard():
    try:
        run = BenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        st.error(f"Cannot read this run: {exc}")
        return
    st.caption(f"Run {run.id} · {'MOCK / offline demo' if run.config.mock else 'Live API results'} · {len(run.benchmark.cases)} scenarios")
    if run.config.mock:
        st.warning("Synthetic demo judgments — these are not evidence of model performance.")
    judges = st.multiselect("Judge models", run.config.judges, default=run.config.judges, key=f"judges-{run.id}")
    observations = [o for o in run.observations if o.judge in judges]
    filtered = run.model_copy(update={"observations": observations,
        "config": run.config.model_copy(update={"judges": judges})})
    rows = statistics(filtered.model_dump_json())
    def rate(name, judge=None):
        selected = [r for r in rows if r["metric"] == name and (judge is None or r["judge"] == judge)]
        n = sum(r["n"] for r in selected)
        return sum(r["numerator"] for r in selected) / n if n else None

    valid = sum(o.status == Status.VALID for o in observations)
    latency = p95(observations)
    usage = [o.completion for o in observations if o.completion and
             o.completion.prompt_tokens is not None and o.completion.completion_tokens is not None]
    tokens = sum(c.prompt_tokens + c.completion_tokens for c in usage)
    cards = [("Saved judgments", f"{len(observations):,}", "Latest outcome per comparison, including errors."),
             ("Valid response rate", percent(valid / len(observations) if observations else None), "Valid verdicts / saved judgments; not answer accuracy."),
             ("Position consistency", percent(rate("position consistency")), "Same winner after swapping order, among complete valid pairs."),
             ("P95 latency", f"{latency:.2f}s" if latency is not None else "—", "95th percentile of recorded completion latency."),
             ("Recorded tokens", f"{tokens:,}" if usage else "—", "Input + output tokens with complete usage metadata; excludes generation and earlier failed attempts."),
             ("Decisive flip rate", percent(rate("decisive flip rate")), "Changed winner after swapping order, excluding ties.")]
    for column, (label, value, help_text) in zip(st.columns(6), cards):
        with column.container(border=True):
            st.metric(label, value, help=help_text)
    st.caption(f"Token coverage: {len(usage)} / {len(observations)} judgments. Rates pool eligible comparisons across selected judges. Missing measurements display —.")

    left, right = st.columns([1.4, 1])
    with left.container(border=True):
        st.subheader("Valid responses over time")
        figure = go.Figure()
        for judge in judges:
            ordered = sorted((o for o in observations if o.judge == judge), key=lambda o: o.created)
            successes = 0
            rates = []
            for i, o in enumerate(ordered, 1):
                successes += o.status == Status.VALID
                rates.append(successes / i)
            figure.add_trace(go.Scatter(x=[o.created for o in ordered], y=rates, name=judge,
                                       mode="lines+markers"))
        figure.update_yaxes(range=[0, 1.05], tickformat=".0%")
        chart(figure)
        st.caption("Cumulative validity of current saved outcomes; this is not a quality score or a history of retries.")
    with right.container(border=True):
        st.subheader("Judgments by experiment")
        counts = Counter(o.bias.value for o in observations)
        figure = go.Figure(go.Bar(x=list(counts.values()), y=list(counts), orientation="h",
                                 marker_color="#9278ed", text=list(counts.values()), textposition="auto"))
        chart(figure)
        st.caption("Includes errors. Self-preference reuses position comparisons, so it is not counted twice.")

    left, middle, right = st.columns([1.2, 1.2, 1])
    with left.container(border=True):
        st.subheader("Model comparison")
        comparison = []
        for judge in judges:
            items = [o for o in observations if o.judge == judge]
            seconds = p95(items)
            comparison.append({"Judge": judge, "Judgments": len(items),
                "Consistency": percent(rate("position consistency", judge)),
                "Flip rate": percent(rate("decisive flip rate", judge)),
                "Repeatability": percent(rate("repeatability", judge)),
                "P95": f"{seconds:.2f}s" if seconds is not None else "—"})
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.caption("Consistency needs both answer orders; repeatability needs multiple valid trials.")
    with middle.container(border=True):
        st.subheader("Preference profile")
        names = ["first-position preference", "own-answer preference",
                 "reviewed verbosity preference", "reviewed style preference"]
        figure = go.Figure()
        for judge in judges:
            available = [r for r in rows if r["judge"] == judge and r["metric"] in names and r["n"]]
            figure.add_trace(go.Bar(name=judge, x=[r["value"] for r in available],
                y=[r["metric"].replace(" preference", "") for r in available], orientation="h",
                customdata=[r["n"] for r in available], hovertemplate="%{y}: %{x:.1%}<br>n=%{customdata}<extra>%{fullData.name}</extra>"))
        figure.update_layout(barmode="group")
        figure.update_xaxes(range=[0, 1], tickformat=".0%")
        chart(figure)
        st.caption("Unreviewed variants are excluded. Preference alone does not prove bias: answer quality can differ.")
    with right.container(border=True):
        st.subheader("Run notices")
        errors = sorted((o for o in observations if o.status != Status.VALID), key=lambda o: o.created, reverse=True)
        if errors:
            for o in errors[:3]:
                st.warning(f"{o.judge} · {o.case_id}\n\n{o.error or o.status.value}")
                st.caption(o.created.strftime("%Y-%m-%d %H:%M:%S UTC"))
        else:
            st.info("No errors in the selected saved outcomes.")
        unreviewed = sum(t.review != Review.APPROVED for c in run.benchmark.cases for t in c.transformations)
        if unreviewed:
            st.caption(f"{unreviewed} transformations await review on the Benchmark page.")
        st.caption(f"{len(run.failed_attempts)} earlier failed attempts retained in this run's history.")

    with st.expander("Metric definitions, sample sizes and 95% confidence intervals"):
        st.caption("Intervals bootstrap scenarios, keeping observations from the same scenario together. Fewer than two scenarios cannot provide an interval.")
        st.dataframe([{**r, "ci95": str(r["ci95"])} for r in rows], hide_index=True)
    st.subheader("Inspect the underlying judgments")
    case_ids = list(dict.fromkeys(o.case_id for o in observations))
    if case_ids:
        case_id = st.selectbox("Scenario", case_ids, key=f"inspect-{run.id}")
        case = next(c for c in run.benchmark.cases if c.id == case_id)
        st.write(case.prompt)
        for col, candidate in zip(st.columns(2), case.candidates):
            with col.container(border=True):
                st.markdown(f"**{candidate.generator}**")
                st.write(candidate.text)
        st.dataframe([{"Judge": o.judge, "Experiment": o.bias.value, "Target": o.target,
            "Trial": o.trial + 1, "Order": " → ".join(o.order), "Status": o.status.value,
            "Winner": o.winner, "Reason": o.decision.reason if o.decision else o.error}
            for o in observations if o.case_id == case_id], hide_index=True)
        st.caption("Position: c0/c1 are the saved answers above. Verbosity/style: c0 is the target's original, c1 its transformed answer. Inspect transformations on the Benchmark page.")
    st.download_button("Download full run JSON", run.model_dump_json(indent=2), f"{run.id}.json", "application/json")


dashboard()
