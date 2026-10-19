"""The saved experiment format. Model names stay strings to allow new providers."""
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


class Bias(StrEnum):
    POSITION = "position"
    VERBOSITY = "verbosity"
    STYLE = "style"
    SELF_PREFERENCE = "self-preference"


class Variant(StrEnum):
    ORIGINAL = "original"
    VERBOSE = "verbose"
    STYLED = "styled"


class Verdict(StrEnum):
    A = "A"
    B = "B"
    TIE = "Tie"


class Movie(BaseModel):
    id: int
    title: str
    genres: list[str]
    runtime: float = Field(gt=0)
    overview: str


class Case(BaseModel):
    source: str = "TMDB 5000"
    request: str = Field(min_length=1)
    movies: list[Movie] = Field(min_length=2)


class Answer(BaseModel):
    model: str
    text: str = Field(min_length=1)
    verbose: str = ""
    styled: str = ""


class Decision(BaseModel):
    verdict: Verdict
    reason: str = Field(min_length=1)


class Judgment(BaseModel):
    judge: str
    variant: Variant
    target: int = Field(ge=0, le=1)
    order: tuple[int, int]
    decision: Decision
    raw: str

    @field_validator("order")
    @classmethod
    def valid_order(cls, value):
        if value not in [(0, 1), (1, 0)]:
            raise ValueError("Order must contain each answer exactly once")
        return value

    @property
    def winner(self) -> int | None:
        if self.decision.verdict == Verdict.TIE:
            return None
        return self.order[0 if self.decision.verdict == Verdict.A else 1]


class Run(BaseModel):
    version: str = "1"
    id: str = Field(default_factory=lambda: uuid4().hex)
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    case: Case
    answers: tuple[Answer, Answer]
    judges: list[str]
    biases: list[Bias]
    rubric: str
    temperature: float = 0
    judgments: list[Judgment] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_run(self):
        if self.version != "1":
            raise ValueError("Unsupported experiment version")
        if self.answers[0].model == self.answers[1].model:
            raise ValueError("Choose two different answer models")
        keys = [(j.judge, j.variant, j.target, j.order) for j in self.judgments]
        if len(keys) != len(set(keys)) or any(j.judge not in self.judges for j in self.judgments):
            raise ValueError("Duplicate judgments or unknown judge")
        return self
