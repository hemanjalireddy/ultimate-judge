import json

import httpx
import pytest
from groq import Groq

from provider import InvalidResponse, ProviderError, parse_decision, request
from provider import RateLimitError


def client_for(statuses, seen):
    def handle(req):
        seen.append(json.loads(req.content))
        status = statuses.pop(0)
        body = {"id": "test-response", "object": "chat.completion", "created": 0, "model": "test",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"verdict":"A","reason":"fit"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        return httpx.Response(status, json=body if status == 200 else {"error": {"message": "DO NOT LOG SECRET", "type": "test"}},
                              headers={"retry-after": "2"})
    return Groq(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(handle)))


def test_backoff_and_request_parameters():
    seen, sleeps = [], []
    result = request(client_for([429, 503, 200], seen), "test", "Return JSON", {}, json_mode=True, sleep=sleeps.append)
    assert result.attempts == 3 and len(seen) == 3
    assert 2 <= sleeps[0] <= 3 and 2 <= sleeps[1] <= 3
    assert result.prompt_tokens == 10 and result.completion_tokens == 5
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["temperature"] == 0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_errors_never_retry(status):
    seen = []
    with pytest.raises(ProviderError) as error:
        request(client_for([status], seen), "test", "system", {})
    assert len(seen) == 1
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("text", ['{}', 'not json', '{"verdict":"C","reason":"why"}', '{"verdict":"A","reason":""}'])
def test_malformed_decisions(text):
    with pytest.raises(InvalidResponse):
        parse_decision(text)


def test_exhausted_retries():
    seen = []
    with pytest.raises(ProviderError):
        request(client_for([500, 500, 500], seen), "test", "system", {}, sleep=lambda _: None)
    assert len(seen) == 3


def test_rate_limit_retries_same_payload_and_reports_wait():
    seen,sleeps,notices=[],[],[]
    result=request(client_for([429,200],seen),'test','Return JSON',{'A':'same answer'},
                   sleep=sleeps.append,notice=notices.append)
    assert seen[0]==seen[1] and result.attempts==2
    assert sum(sleeps)>=2 and notices


def test_exhausted_rate_limit_keeps_retry_time():
    with pytest.raises(RateLimitError) as caught:
        request(client_for([429],[]),'test','system',{},max_attempts=1)
    assert caught.value.retry_after>=2
