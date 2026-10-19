"""Case-cluster bootstrap; all rates expose numerator, denominator and exclusions."""
import itertools
import random
from collections import defaultdict

from benchmark_models import BenchmarkRun, Reference, Review
from models import Bias, Verdict
from provider import Status


def proportion(samples: list[tuple[str, float]], *, seed: int = 42, draws: int = 1000) -> dict:
    groups = defaultdict(list)
    for case, value in samples:
        groups[case].append(value)
    n = len(samples)
    result = {"numerator": sum(v for _, v in samples), "n": n, "cases": len(groups),
              "value": sum(v for _, v in samples) / n if n else None, "ci95": None}
    if len(groups) >= 2:
        rng = random.Random(seed)
        ids = sorted(groups)
        estimates = []
        for _ in range(draws):
            selected = [v for key in rng.choices(ids, k=len(ids)) for v in groups[key]]
            estimates.append(sum(selected) / len(selected))
        estimates.sort()
        result["ci95"] = [estimates[int(draws * .025)], estimates[min(draws - 1, int(draws * .975))]]
    return result


def summary(run: BenchmarkRun) -> list[dict]:
    cases = {c.id: c for c in run.benchmark.cases}
    rows = []
    for judge in run.config.judges:
        observations = [o for o in run.observations if o.judge == judge]
        valid = [o for o in observations if o.status == Status.VALID]
        baseline = [o for o in valid if o.bias == Bias.POSITION]
        samples = defaultdict(list)
        for o in observations:
            samples["invalid output rate"].append((o.case_id, float(o.status == Status.INVALID)))
            samples["API failure rate"].append((o.case_id, float(o.status == Status.API_ERROR)))
        for o in valid:
            samples["tie rate"].append((o.case_id, float(o.winner == "Tie")))
        for o in baseline:
            case = cases[o.case_id]
            if o.winner != "Tie":
                samples["first-position preference"].append((o.case_id, float(o.decision.verdict == Verdict.A)))
                generators = [c.generator for c in case.candidates]
                if Bias.SELF_PREFERENCE in run.config.biases and generators.count(judge) == 1:
                    winning_model = generators[0 if o.winner == "c0" else 1]
                    samples["own-answer preference"].append((o.case_id, float(winning_model == judge)))
            if case.reference_review == Review.APPROVED and case.ground_truth != Reference.UNCERTAIN:
                samples["reference agreement"].append((o.case_id, float(o.winner == case.ground_truth)))
        pairs = defaultdict(list)
        repeats = defaultdict(list)
        for o in baseline:
            pairs[(o.case_id, o.trial)].append(o)
        for o in valid:
            repeats[(o.case_id, o.bias, o.target, o.order)].append(o)
        for pair in pairs.values():
            if len(pair) != 2 or pair[0].order == pair[1].order:
                continue
            a, b = pair
            samples["position consistency"].append((a.case_id, float(a.winner == b.winner)))
            samples["tie transition rate"].append((a.case_id, float((a.winner == "Tie") != (b.winner == "Tie"))))
            if a.winner != "Tie" and b.winner != "Tie":
                samples["decisive flip rate"].append((a.case_id, float(a.winner != b.winner)))
        for group in repeats.values():
            for a, b in itertools.combinations(group, 2):
                samples["repeatability"].append((a.case_id, float(a.winner == b.winner)))
        for o in valid:
            if o.bias not in (Bias.VERBOSITY, Bias.STYLE):
                continue
            transform = next(t for t in cases[o.case_id].transformations if t.bias == o.bias and t.candidate_id == o.target)
            if transform.review == Review.APPROVED:
                samples[f"reviewed {o.bias.value} preference"].append((o.case_id, float(o.winner == "c1")))
        names = ["position consistency", "decisive flip rate", "tie transition rate", "repeatability",
                 "first-position preference", "reviewed verbosity preference", "reviewed style preference",
                 "own-answer preference", "reference agreement", "tie rate", "invalid output rate", "API failure rate"]
        for name in names:
            rows.append({"judge": judge, "metric": name, **proportion(samples[name], seed=run.config.seed),
                         "observations": len(observations), "valid_observations": len(valid)})
    return rows


def disagreements(run: BenchmarkRun) -> list[dict]:
    groups = defaultdict(dict)
    for o in run.observations:
        if o.status == Status.VALID:
            groups[(o.case_id, o.bias, o.target, o.trial, o.order)][o.judge] = o.winner
    return [{"case": key[0], "bias": key[1].value, "target": key[2], "trial": key[3], "order": str(key[4]), **votes}
            for key, votes in groups.items() if len(set(votes.values())) > 1]


def resource_usage(run: BenchmarkRun) -> list[dict]:
    rows = []
    for judge in run.config.judges:
        outputs = [o.completion for o in run.observations if o.judge == judge and o.completion]
        rows.append({"judge": judge, "responses_with_metadata": len(outputs),
                     "mean_latency_seconds": sum(c.latency_seconds for c in outputs) / len(outputs) if outputs else None,
                     "prompt_tokens": sum(c.prompt_tokens for c in outputs if c.prompt_tokens is not None),
                     "completion_tokens": sum(c.completion_tokens for c in outputs if c.completion_tokens is not None),
                     "responses_with_usage": sum(c.prompt_tokens is not None and c.completion_tokens is not None for c in outputs)})
    return rows


def generator_matrix(run: BenchmarkRun) -> list[dict]:
    cases = {c.id:c for c in run.benchmark.cases}
    groups = defaultdict(list)
    for o in run.observations:
        if o.bias != Bias.POSITION or o.status != Status.VALID or o.winner == "Tie":
            continue
        case = cases[o.case_id]
        for candidate in case.candidates:
            quality = ("unreviewed" if case.reference_review != Review.APPROVED else
                       "reference winner" if case.ground_truth == candidate.id else
                       "reference loser" if case.ground_truth in (Reference.FIRST, Reference.SECOND) else case.ground_truth.value)
            groups[(o.judge,candidate.generator,quality)].append((o.case_id, float(o.winner == candidate.id)))
    return [{"judge":key[0],"generator":key[1],"quality stratum":key[2], **proportion(values, seed=run.config.seed)}
            for key,values in groups.items()]
