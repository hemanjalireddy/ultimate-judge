"""Build frozen real-TMDB cases; demo fixtures are explicitly synthetic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from benchmark_models import (Benchmark, BenchmarkCase, Candidate, Constraints,
                              Facts, Transformation)
from models import Bias
from questions import Scenario, Scenarios

# Six explicit question families, five different movie sets per family.
SCENARIOS = [
    ("sci-fi-runtime", "Recommend a science-fiction movie under 120 minutes.", Constraints(genre="Science Fiction", max_runtime=120)),
    ("recent-thriller", "Recommend a thriller released in or after 2015 with a rating of at least 6.5.", Constraints(genre="Thriller", min_year=2015, min_rating=6.5)),
    ("short-animation", "Recommend an animated movie under 100 minutes.", Constraints(genre="Animation", max_runtime=100)),
    ("popular-comedy", "Recommend a comedy rated at least 7 with at least 1000 votes.", Constraints(genre="Comedy", min_rating=7, min_votes=1000)),
    ("modern-drama", "Recommend a drama from 2010 or later under 120 minutes.", Constraints(genre="Drama", min_year=2010, max_runtime=120)),
    ("family-runtime", "Recommend a family movie under 100 minutes rated at least 6.5.", Constraints(genre="Family", max_runtime=100, min_rating=6.5)),
]


def variants(candidate: Candidate) -> list[Transformation]:
    return [
        Transformation(bias=Bias.VERBOSITY, candidate_id=candidate.id,
                       text=candidate.text + "\n\n" + candidate.text, kind="exact-repetition"),
        Transformation(bias=Bias.STYLE, candidate_id=candidate.id,
                       text="> " + candidate.text.replace("\n", "\n> "), kind="blockquote-formatting"),
    ]


def build(csv_path: Path) -> Scenarios:
    raw = csv_path.read_bytes()
    movies = []
    for row in csv.DictReader(raw.decode("utf-8-sig").splitlines()):
        if not row["runtime"] or float(row["runtime"]) <= 0 or not row["release_date"]:
            continue
        movies.append(Facts(id=int(row["id"]), title=row["title"],
            genres=[g["name"] for g in json.loads(row["genres"])], runtime=float(row["runtime"]),
            year=int(row["release_date"][:4]), rating=float(row["vote_average"]), votes=int(row["vote_count"])))
    movies.sort(key=lambda m: m.id)
    cases = []
    for prefix, prompt, constraints in SCENARIOS:
        passing = [m for m in movies if all(v is True for v in constraints.check(m).values())]
        failing = [m for m in movies if any(v is False for v in constraints.check(m).values())]
        if len(passing) < 5 or len(failing) < 5:
            raise ValueError(f"Insufficient movies for {prefix}")
        for i in range(5):
            facts = [passing[i], failing[i]]
            # Alternate canonical locations of constraint-compliant answers.
            if i % 2:
                facts.reverse()
            cases.append(Scenario(id=f"{prefix}-{i+1:02}", prompt=prompt + " Choose only from the supplied movies.",
                movie_facts=facts, criteria=["Meet explicit request constraints", "Use supplied facts accurately", "Explain relevant fit"],
                constraints=constraints))
    return Scenarios(snapshot="sha256:" + hashlib.sha256(raw).hexdigest(),
        source="https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata", cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/scenarios.json"))
    args = parser.parse_args()
    benchmark = build(args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(benchmark.model_dump_json(indent=2), encoding="utf-8")
    print(f"Saved {len(benchmark.cases)} cases; {benchmark.snapshot}")
