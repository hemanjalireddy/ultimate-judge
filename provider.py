"""Groq boundary: bounded retries, safe errors, validated response metadata."""
import json
import os
import random
import time
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from enum import StrEnum
from typing import Callable

from groq import APIConnectionError, APIStatusError, Groq
from pydantic import BaseModel, Field, ValidationError

from models import Decision


class Status(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    API_ERROR = "api_error"


class Completion(BaseModel):
    text: str
    response_id: str | None = None
    latency_seconds: float = Field(ge=0)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    attempts: int = 1


class ProviderError(RuntimeError):
    """Only safe categories are exposed; never store provider exception bodies."""


class InvalidResponse(ProviderError):
    pass


class RateLimitError(ProviderError):
    def __init__(self, seconds: float):
        self.retry_after = seconds
        super().__init__(f"Rate limited. Retry this comparison in approximately {math.ceil(seconds)} seconds.")


def retry_seconds(value: str | None) -> float:
    try:
        return max(1, float(value))
    except (ValueError, TypeError):
        try:
            return max(1, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (ValueError, TypeError, OverflowError):
            return 60


def wait_for(seconds: float, sleep: Callable[[float], None], notice: Callable[[str], None] | None) -> None:
    while seconds > 0:
        if notice:
            notice(f"Waiting {math.ceil(seconds)} seconds before the next attempt; saved answers are unchanged.")
        step = min(seconds, 10)
        sleep(step)
        seconds -= step


class Pacer:
    """Conservative per-model spacing; provider limits remain authoritative."""
    def __init__(self):
        self.next_call: dict[str, float] = {}

    def before(self, model: str, estimated_tokens: int, sleep, notice) -> None:
        wait_for(max(0, self.next_call.get(model, 0) - time.monotonic()), sleep, notice)
        self.next_call[model] = time.monotonic() + max(2.1, 60 * estimated_tokens / 7000)


def make_client() -> Groq:
    if not os.getenv("GROQ_API_KEY"):
        raise ProviderError("Set GROQ_API_KEY in the environment.")
    return Groq(timeout=60, max_retries=0)


def request(client: Groq, model: str, system: str, payload: dict, *,
            json_mode: bool = False, max_attempts: int = 3,
            sleep: Callable[[float], None] = time.sleep,
            notice: Callable[[str], None] | None = None,
            pacer: Pacer | None = None) -> Completion:
    start = time.perf_counter()
    # Disable nested SDK retries even for callers using the legacy interface.
    client = client.with_options(max_retries=0)
    for attempt in range(max_attempts):
        if pacer:
            estimated_tokens = math.ceil((len(system) + len(json.dumps(payload))) / 3) + 1024
            pacer.before(model, estimated_tokens, sleep, notice)
        try:
            response = client.chat.completions.create(
                model=model, temperature=0, max_completion_tokens=4096,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(payload)}],
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
            )
            break
        except (APIStatusError, APIConnectionError) as error:
            code = getattr(error, "status_code", None)
            if code == 429:
                delay = retry_seconds(error.response.headers.get("retry-after")) + 1
                if delay > 300 or attempt + 1 == max_attempts:
                    raise RateLimitError(delay) from None
                wait_for(delay, sleep, notice)
                continue
            retryable = code is None or code in (408, 409, 429) or code >= 500
            if not retryable or attempt + 1 == max_attempts:
                raise ProviderError(f"Groq request failed ({code or 'connection'}).") from None
            delay = min(2 ** attempt + random.uniform(0, 1), 30)
            header = getattr(error, "response", None)
            try:
                retry_after = float(header.headers.get("retry-after", "0")) if header else 0
            except ValueError:
                retry_after = 0
            if retry_after > 300:
                raise ProviderError("Provider unavailable; resume later.") from None
            wait_for(max(delay, retry_after), sleep, notice)
    if not getattr(response, "choices", None):
        raise InvalidResponse("Groq returned no choices.")
    choice = response.choices[0]
    content = getattr(getattr(choice, "message", None), "content", None)
    if choice.finish_reason != "stop" or not isinstance(content, str) or not content.strip():
        raise InvalidResponse("Groq returned an incomplete or empty response.")
    usage = response.usage
    return Completion(text=choice.message.content, response_id=response.id,
        latency_seconds=time.perf_counter() - start, attempts=attempt + 1,
        prompt_tokens=usage.prompt_tokens if usage else None,
        completion_tokens=usage.completion_tokens if usage else None)


def parse_decision(text: str) -> Decision:
    try:
        return Decision.model_validate_json(text)
    except ValidationError:
        raise InvalidResponse("Judge output must contain verdict A, B or Tie and a nonempty reason.") from None


def verify_models(models: list[str]) -> list[str]:
    try:
        available = {model.id for model in make_client().models.list().data}
    except (APIStatusError, APIConnectionError):
        raise ProviderError("Could not list Groq models; check credentials and connectivity.") from None
    missing = set(models) - available
    if missing:
        raise ProviderError(f"Unavailable model IDs: {', '.join(sorted(missing))}")
    return sorted(set(models))
