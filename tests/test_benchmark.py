from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark import conditions, execute, jobs
from benchmark_models import Benchmark, BenchmarkRun, Config, Observation, Review
from models import Bias, Decision, Verdict, Run
from provider import Status


@pytest.fixture
def config():
    return Config.model_validate_json(Path('configs/demo.json').read_text())


def test_offline_resume_and_roundtrip(config, tmp_path):
    run = execute(config, tmp_path)
    assert len(run.observations) == 120  # 2 cases x 2 judges x 5 conditions x 3 trials x 2 orders
    assert all(o.completion.response_id.startswith('mock-') for o in run.observations)
    assert execute(config, tmp_path) == run
    assert BenchmarkRun.model_validate_json(run.model_dump_json()) == run
    assert 'GROQ_API_KEY' not in run.model_dump_json()


def test_budget_and_lock(config, tmp_path):
    config.request_budget = 2
    run = execute(config, tmp_path)
    assert len(run.observations) == 2
    assert len(execute(config, tmp_path).observations) == 2
    config.request_budget = 4
    extended = execute(config, tmp_path)
    assert extended.id == run.id and len(extended.observations) == 4
    (tmp_path / f'{run.id}.lock').touch()
    with pytest.raises(FileExistsError):
        execute(config, tmp_path)


@pytest.mark.parametrize('order', [('c0', 'c1'), ('c1', 'c0')])
@pytest.mark.parametrize('verdict', list(Verdict))
def test_canonical_mapping(order, verdict):
    obs = Observation(id='x', case_id='x', judge='x', bias=Bias.POSITION, trial=0, order=order,
                      status=Status.VALID, decision=Decision(verdict=verdict, reason='test'))
    expected = 'Tie' if verdict == Verdict.TIE else order[0 if verdict == Verdict.A else 1]
    assert obs.winner == expected


def test_review_schema_and_unknown_version(config):
    benchmark = Benchmark.model_validate_json(Path(config.benchmark).read_text())
    raw = benchmark.model_dump()
    raw['cases'][0]['transformations'][0]['review'] = Review.APPROVED
    with pytest.raises(ValidationError):
        Benchmark.model_validate(raw)
    raw['version'] = 'future-v99'
    with pytest.raises(ValidationError):
        Benchmark.model_validate(raw)


def test_transformations_and_seed(config):
    benchmark = Benchmark.model_validate_json(Path(config.benchmark).read_text())
    assert list(jobs(benchmark, config)) == list(jobs(benchmark, config))
    case = benchmark.cases[0]
    for bias, target, texts in conditions(case, config):
        if bias == Bias.VERBOSITY:
            assert texts[1] == texts[0] + '\n\n' + texts[0]
        if bias == Bias.STYLE:
            assert texts[1] == '> ' + texts[0]


def test_legacy_run_compatible():
    payload = {'case': {'request': 'pick', 'movies': [{'id': i, 'title': str(i), 'genres': [], 'runtime': 90, 'overview': 'story'} for i in (1,2)]},
               'answers': [{'model': 'x', 'text': 'one'}, {'model': 'y', 'text': 'two'}], 'judges': ['x'], 'biases': ['position'], 'rubric': 'fit'}
    run = Run.model_validate(payload)
    assert Run.model_validate_json(run.model_dump_json()) == run


def test_invalid_is_not_tie():
    obs = Observation(id='x', case_id='x', judge='x', bias=Bias.POSITION, trial=0, order=('c0','c1'), status=Status.INVALID)
    assert obs.winner is None
    with pytest.raises(ValidationError):
        Observation.model_validate({**obs.model_dump(), 'decision': {'verdict':'Tie','reason':'bad'}})
