"""Question uploads and fact lookup. No generated answers belong in this format."""
import csv
import hashlib
import io
import json
import re
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, model_validator

from benchmark_models import Candidate, Constraints, Facts


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    movie_ids: list[int] = Field(default_factory=list)
    criteria: list[str] = Field(default_factory=lambda: [
        "Meet the user's request", "Use the supplied facts accurately", "Give a relevant explanation"])


class Scenario(BaseModel):
    id: str
    prompt: str
    movie_facts: list[Facts] = Field(min_length=2)
    criteria: list[str]
    constraints: Constraints = Field(default_factory=Constraints)


class Scenarios(BaseModel):
    version: str = "scenarios-v1"
    snapshot: str
    source: str
    cases: list[Scenario] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cases(self):
        if self.version != "scenarios-v1":
            raise ValueError("Unsupported scenario version")
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Question IDs must be unique")
        return self


class GenerationProgress(BaseModel):
    source_hash: str
    models: tuple[str, str]
    answers: dict[str, list[Candidate]] = Field(default_factory=dict)


def parse_questions(content: bytes, filename: str) -> list[Question]:
    text = content.decode("utf-8-sig")
    if Path(filename).suffix.lower() == ".txt":
        rows = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if isinstance(rows, dict):
            rows = rows.get("questions")
        if not isinstance(rows, list):
            raise ValueError('JSON must be a list or an object containing a "questions" list')
    if any(not isinstance(row, (str, dict)) for row in rows):
        raise ValueError("Each question must be text or a JSON object")
    questions = [Question.model_validate({"id": f"question-{i+1:03}", "prompt": row} if isinstance(row, str)
                 else {"id": f"question-{i+1:03}", **row}) for i, row in enumerate(rows)]
    if not questions or len(questions) > 100:
        raise ValueError("Upload between 1 and 100 questions")
    if len({q.id for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique")
    return questions


def load_catalog(raw: bytes) -> dict[int, Facts]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    required = {"id", "title", "genres", "runtime", "release_date", "vote_average", "vote_count"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Upload tmdb_5000_movies.csv with movie metadata columns")
    movies = {}
    for row in reader:
        if not row["runtime"] or float(row["runtime"]) <= 0 or not row["release_date"]:
            continue
        movie = Facts(id=int(row["id"]), title=row["title"],
            genres=[g["name"] for g in json.loads(row["genres"])], runtime=float(row["runtime"]),
            year=int(row["release_date"][:4]), rating=float(row["vote_average"]), votes=int(row["vote_count"]))
        movies[movie.id] = movie
    if not movies:
        raise ValueError("Catalog has no usable movies")
    return movies


def automatic_movie_ids(question: Question, catalog: dict[int, Facts]) -> list[int]:
    """Small deterministic genre/title lookup, not a natural-language constraint solver."""
    prompt = re.sub(r"[^a-z0-9 ]", " ", question.prompt.lower()).replace("sci fi", "science fiction")
    def score(movie: Facts) -> tuple[int, int]:
        title = re.sub(r"[^a-z0-9 ]", " ", movie.title.lower())
        genres = sum(g.lower() in prompt for g in movie.genres)
        title_match = len(title) >= 4 and title in prompt
        return (int(title_match) * 100 + genres, -movie.id)
    return [m.id for m in sorted(catalog.values(), key=score, reverse=True)[:6]]


def make_scenarios(questions: list[Question], raw_catalog: bytes, shared_ids: list[int] | None = None) -> Scenarios:
    catalog = load_catalog(raw_catalog)
    cases = []
    for question in questions:
        ids = question.movie_ids or shared_ids or automatic_movie_ids(question, catalog)
        if not 2 <= len(ids) <= 10 or len(set(ids)) != len(ids):
            raise ValueError(f"{question.id}: choose 2–10 distinct movie options")
        missing = set(ids) - catalog.keys()
        if missing:
            raise ValueError(f"{question.id}: movie IDs not found in catalog: {sorted(missing)}")
        cases.append(Scenario(id=question.id, prompt=question.prompt,
                              movie_facts=[catalog[i] for i in ids], criteria=question.criteria))
    return Scenarios(snapshot="sha256:" + hashlib.sha256(raw_catalog).hexdigest(),
                     source="User questions grounded in local TMDB CSV", cases=cases)
