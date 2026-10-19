"""Exercise the visible upload -> Run test -> Run judge sequence without API calls."""
from pathlib import Path
from types import SimpleNamespace
import shutil

from streamlit.testing.v1 import AppTest

import benchmark as engine
from benchmark_models import Benchmark, BenchmarkCase, BenchmarkRun, Candidate, Observation
from models import Bias, Decision, Verdict
from provider import Completion, Status
from questions import Scenarios
from runner import MODELS


def test_main_question_flow(tmp_path, monkeypatch):
    root = Path.cwd()
    shutil.copy(root/'app.py', tmp_path/'app.py')
    (tmp_path/'pages').mkdir()
    (tmp_path/'pages/1_Benchmark.py').write_text('')
    (tmp_path/'examples').mkdir()
    shutil.copy(root/'examples/30_questions.txt', tmp_path/'examples/30_questions.txt')
    (tmp_path/'data').mkdir()
    (tmp_path/'data/tmdb_5000_movies.csv').write_text(
        'id,title,genres,runtime,release_date,vote_average,vote_count\n'
        '1,One,[],90,2020-01-01,7,100\n2,Two,[],100,2021-01-01,8,200\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GROQ_API_KEY', 'test-only')
    labels = []
    def upload(label, **kwargs):
        labels.append(label)
        return SimpleNamespace(name='questions.txt', getvalue=lambda: b'Which supplied movie is shorter?')
    monkeypatch.setattr('streamlit.file_uploader', upload)

    def generate(source, destination, models):
        scenarios=Scenarios.model_validate_json(source.read_text())
        cases=[BenchmarkCase(**case.model_dump(), candidates=tuple(
            Candidate(id=f'c{i}', generator=model, text=f'Answer from {model}',
                      generation=Completion(text='answer', latency_seconds=0))
            for i,model in enumerate(models))) for case in scenarios.cases]
        engine.write_json(destination, Benchmark(snapshot=scenarios.snapshot,source='test',cases=cases))
    monkeypatch.setattr(engine,'generate_candidates',generate)
    def judge(config, **kwargs):
        frozen=Benchmark.model_validate_json(Path(config.benchmark).read_text())
        return BenchmarkRun(id='test',config=config,benchmark=frozen,observations=[Observation(
            id='one',case_id=frozen.cases[0].id,judge=MODELS[0],bias=Bias.POSITION,trial=0,
            order=('c0','c1'),status=Status.VALID,decision=Decision(verdict=Verdict.A,reason='Test judgment'))])
    monkeypatch.setattr(engine,'execute',judge)
    app=AppTest.from_file(tmp_path/'app.py',default_timeout=30).run()
    assert not app.exception
    assert all(label.startswith('Questions file') for label in labels)
    next(b for b in app.button if b.label=='Run test').click().run()
    assert not app.exception
    assert all(any(f'Answer from {model}' in m.value for m in app.markdown) for model in MODELS)
    next(b for b in app.button if b.label=='Run judge').click().run()
    assert not app.exception
    assert any(s.value=='4. Results' for s in app.subheader)
    # Changing answer-model selection must not leave stale answers or verdicts visible.
    next(m for m in app.multiselect if m.label=='Answer models — choose two').set_value([MODELS[0]]).run()
    assert not any(s.value=='4. Results' for s in app.subheader)
