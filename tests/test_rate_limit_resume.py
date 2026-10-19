from datetime import datetime, timedelta, timezone
from pathlib import Path

import benchmark as engine
from benchmark_models import Benchmark, BenchmarkRun, Config
from provider import Completion, RateLimitError, Status, retry_seconds


def test_resume_retries_failed_comparison_and_preserves_history(tmp_path, monkeypatch):
    source=Benchmark.model_validate_json(Path('examples/demo-benchmark.json').read_text())
    source.mock=False
    source.cases=source.cases[:1]
    for i, candidate in enumerate(source.cases[0].candidates):
        candidate.generator=engine.MODELS[i]
        candidate.generation=Completion(text=candidate.text, latency_seconds=0)
    source_path=tmp_path/'source.json'
    engine.write_json(source_path,source)
    config=Config(benchmark=str(source_path),judges=engine.MODELS,trials=1,mock=False,request_budget=100)
    monkeypatch.setattr(engine,'make_client',lambda:object())
    calls=[]
    def request(*args,**kwargs):
        calls.append(args[3])
        if len(calls)==2:
            raise RateLimitError(600)
        return Completion(text='{"verdict":"A","reason":"test"}',latency_seconds=0)
    monkeypatch.setattr(engine,'request',request)
    run=engine.execute(config,tmp_path/'runs')
    assert len(run.observations)==2
    first_id=run.observations[0].id
    failed=run.observations[-1]
    assert failed.status==Status.API_ERROR and failed.retry_at
    notices=[]
    engine.execute(config,tmp_path/'runs',notice=notices.append)
    assert len(calls)==2 and 'Paused' in notices[-1]
    # Simulate the documented reset having passed, without a real sleep or API call.
    failed.retry_at=datetime.now(timezone.utc)-timedelta(seconds=1)
    engine.write_json(tmp_path/'runs'/f'{run.id}.json',run)
    resumed=engine.execute(config,tmp_path/'runs')
    assert calls[1]==calls[2]
    assert sum(o.id==first_id for o in resumed.observations)==1
    assert len(resumed.failed_attempts)==1 and resumed.failed_attempts[0].id==failed.id
    assert all(o.status==Status.VALID for o in resumed.observations)
    assert BenchmarkRun.model_validate_json(resumed.model_dump_json())==resumed


def test_retry_header_defaults_and_seconds():
    assert retry_seconds('120')==120
    assert retry_seconds(None)==60
