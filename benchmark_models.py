"""Versioned benchmark inputs and observations; independent of legacy Run v1."""
from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from models import Bias, Decision
from provider import Completion, Status


class Review(StrEnum):
    UNREVIEWED = "unreviewed"
    APPROVED = "approved"


class Reference(StrEnum):
    FIRST = "c0"
    SECOND = "c1"
    TIE = "Tie"
    UNCERTAIN = "uncertain"


class Facts(BaseModel):
    id: int
    title: str
    genres: list[str]
    runtime: float = Field(gt=0)
    year: int
    rating: float = Field(ge=0, le=10)
    votes: int = Field(ge=0)
    director: str | None = None


class Constraints(BaseModel):
    genre: str | None = None
    max_runtime: float | None = None
    min_year: int | None = None
    min_rating: float | None = None
    min_votes: int | None = None
    director: str | None = None

    def check(self, movie: Facts) -> dict[str, bool | None]:
        checks = {}
        if self.genre is not None:
            checks["genre"] = self.genre in movie.genres
        for name, actual, limit, minimum in [
            ("runtime", movie.runtime, self.max_runtime, False),
            ("year", movie.year, self.min_year, True),
            ("rating", movie.rating, self.min_rating, True),
            ("votes", movie.votes, self.min_votes, True),
        ]:
            if limit is not None:
                checks[name] = actual >= limit if minimum else actual < limit
        if self.director is not None:
            checks["director"] = None if movie.director is None else movie.director == self.director
        return checks


class Candidate(BaseModel):
    id: Literal["c0", "c1"]
    text: str = Field(min_length=1)
    generator: str = "template-v1"
    generation: Completion | None = None


class Transformation(BaseModel):
    bias: Bias
    candidate_id: Literal["c0", "c1"]
    text: str = Field(min_length=1)
    kind: str
    review: Review = Review.UNREVIEWED
    reviewer: str | None = None
    assertion: str | None = None

    @model_validator(mode="after")
    def review_evidence(self):
        if self.bias not in (Bias.VERBOSITY, Bias.STYLE):
            raise ValueError("Only verbosity and style have transformations")
        if self.review == Review.APPROVED and (not self.reviewer or not self.assertion):
            raise ValueError("Approval requires reviewer and equivalence assertion")
        return self


class BenchmarkCase(BaseModel):
    id: str
    prompt: str
    movie_facts: list[Facts] = Field(min_length=2)
    criteria: list[str]
    constraints: Constraints
    candidates: tuple[Candidate, Candidate]
    transformations: list[Transformation] = Field(default_factory=list)
    ground_truth: Reference = Reference.UNCERTAIN
    reference_review: Review = Review.UNREVIEWED
    reference_reviewer: str | None = None

    @model_validator(mode="after")
    def valid_ids(self):
        if [c.id for c in self.candidates] != ["c0", "c1"]:
            raise ValueError("Candidates must be in canonical c0, c1 order")
        keys = [(t.bias, t.candidate_id) for t in self.transformations]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate transformations")
        if self.reference_review == Review.APPROVED and not self.reference_reviewer:
            raise ValueError("A human reference label needs a reviewer")
        return self


class Benchmark(BaseModel):
    version: Literal["benchmark-v1"] = "benchmark-v1"
    snapshot: str
    source: str
    mock: bool = False
    generation_source: str | None = None
    cases: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self):
        if len({c.id for c in self.cases}) != len(self.cases):
            raise ValueError("Duplicate case IDs")
        return self


class Config(BaseModel):
    version: Literal["config-v1"] = "config-v1"
    benchmark: str
    judges: list[str] = Field(min_length=1)
    case_ids: list[str] = Field(default_factory=list)
    biases: list[Bias] = Field(default_factory=lambda: list(Bias))
    trials: int = Field(default=3, ge=1, le=100)
    seed: int = 42
    mock: bool = True
    request_budget: int = Field(default=1000, ge=0)

    @model_validator(mode="after")
    def unique_selections(self):
        if len(set(self.judges)) != len(self.judges) or len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("Selections must be unique")
        return self


class Observation(BaseModel):
    id: str
    case_id: str
    judge: str
    bias: Bias
    target: str | None = None
    trial: int
    order: tuple[str, str]
    status: Status
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decision: Decision | None = None
    completion: Completion | None = None
    error: str | None = None
    retry_at: datetime | None = None

    @model_validator(mode="after")
    def valid_decision(self):
        if set(self.order) != {"c0", "c1"}:
            raise ValueError("Order must contain c0 and c1")
        if (self.status == Status.VALID) != (self.decision is not None):
            raise ValueError("Only valid observations have a decision")
        return self

    @property
    def winner(self) -> str | None:
        if self.decision is None:
            return None
        return {"A": self.order[0], "B": self.order[1], "Tie": "Tie"}[self.decision.verdict]


class BenchmarkRun(BaseModel):
    version: Literal["benchmark-run-v1"] = "benchmark-run-v1"
    id: str
    config: Config
    benchmark: Benchmark
    prompt_version: str = "pairwise-v1"
    system_prompt: str | None = None
    temperature: float = 0
    max_completion_tokens: int = 4096
    requests_used: int = 0
    observations: list[Observation] = Field(default_factory=list)
    failed_attempts: list[Observation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_observations(self):
        cases = {case.id: case for case in self.benchmark.cases}
        seen = set()
        for o in self.observations:
            key = (o.case_id, o.judge, o.bias, o.target, o.trial, o.order)
            if key in seen or o.case_id not in cases or o.judge not in self.config.judges:
                raise ValueError("Duplicate observation or unknown case/judge")
            if o.trial < 0 or o.trial >= self.config.trials:
                raise ValueError("Trial outside configured range")
            if o.bias != Bias.POSITION and not any(t.bias == o.bias and t.candidate_id == o.target for t in cases[o.case_id].transformations):
                raise ValueError("Unknown transformation")
            seen.add(key)
        return self
