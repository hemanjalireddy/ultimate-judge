from pathlib import Path

import pytest

import benchmark as engine
from benchmark_models import Benchmark, Config
from provider import Completion, ProviderError, Status
from questions import GenerationProgress, Scenario, Scenarios


def test_generation_checkpoint_then_anonymous_judging(tmp_path, monkeypatch):
    source = Benchmark.model_validate_json(Path('examples/demo-benchmark.json').read_text())
    source.mock = False  # Synthetic fixture standing in for real data; never exported as findings.
    source.cases = source.cases[:1]
    input_path, output_path = tmp_path/'source.json', tmp_path/'generated.json'
    source = Scenarios(snapshot=source.snapshot, source=source.source,
                       cases=[Scenario.model_validate(c.model_dump()) for c in source.cases])
    engine.write_json(input_path, source)
    monkeypatch.setattr(engine, 'verify_models', lambda _: None)
    monkeypatch.setattr(engine, 'make_client', lambda: object())
    calls = []

    def generate(*args, **kwargs):
        calls.append(args[1])
        if len(calls) == 2:
            raise ProviderError('temporary fixture failure')
        return Completion(text='A frozen recommendation.', response_id='fixture', latency_seconds=.1)

    monkeypatch.setattr(engine, 'request', generate)
    with pytest.raises(ProviderError):
        engine.generate_candidates(input_path, output_path)
    assert not output_path.exists()
    partial = GenerationProgress.model_validate_json(output_path.with_suffix('.progress.json').read_text())
    assert len(partial.answers[source.cases[0].id]) == 1
    engine.generate_candidates(input_path, output_path)
    assert calls == [engine.MODELS[0], engine.MODELS[1], engine.MODELS[1]]
    engine.generate_candidates(input_path, output_path)
    assert len(calls) == 3  # Completed generation is not repeated.
    with pytest.raises(ValueError,match='checkpoint'):
        engine.generate_candidates(input_path, output_path, list(reversed(engine.MODELS)))

    payloads = []
    def judge(client, model, system, payload, **kwargs):
        payloads.append(payload)
        assert 'generator' not in str(payload) and all(m not in str(payload) for m in engine.MODELS)
        return Completion(text='{"verdict":"Tie","reason":"fixture"}', response_id='fixture', latency_seconds=.2)

    monkeypatch.setattr(engine, 'request', judge)
    config=Config(benchmark=str(output_path), judges=engine.MODELS, trials=1, mock=False, request_budget=100)
    run=engine.execute(config,tmp_path/'runs')
    assert len(run.observations)==20
    assert all(o.status == Status.VALID for o in run.observations)
    assert run.system_prompt == engine.PROMPT
    engine.execute(config,tmp_path/'runs')
    assert len(payloads)==20
    source.cases[0].prompt='A changed question'
    engine.write_json(input_path,source)
    with pytest.raises(ValueError,match='checkpoint'):
        engine.generate_candidates(input_path,output_path)


def test_live_self_preference_rejects_templates(tmp_path):
    source=Benchmark.model_validate_json(Path('examples/demo-benchmark.json').read_text())
    source.mock=False
    engine.write_json(tmp_path/'source.json',source)
    config=Config(benchmark=str(tmp_path/'source.json'),judges=engine.MODELS,mock=False)
    with pytest.raises(ValueError,match='AI-generated'):
        engine.execute(config,tmp_path/'runs')


def test_answer_models_are_user_selected(tmp_path,monkeypatch):
    demo=Benchmark.model_validate_json(Path('examples/demo-benchmark.json').read_text())
    scenarios=Scenarios(snapshot=demo.snapshot,source='test fixture',
                        cases=[Scenario.model_validate(demo.cases[0].model_dump())])
    source,output=tmp_path/'questions.json',tmp_path/'answers.json'
    engine.write_json(source,scenarios)
    selected=['chosen-model-x','chosen-model-y']
    verified,calls=[],[]
    monkeypatch.setattr(engine,'verify_models',lambda models:verified.extend(models))
    monkeypatch.setattr(engine,'make_client',lambda:object())
    def answer(client,model,*args,**kwargs):
        calls.append(model)
        return Completion(text='Answer from '+model,response_id='test',latency_seconds=0)
    monkeypatch.setattr(engine,'request',answer)
    engine.generate_candidates(source,output,selected)
    frozen=Benchmark.model_validate_json(output.read_text())
    assert verified==selected and calls==selected
    assert [c.generator for c in frozen.cases[0].candidates]==selected
