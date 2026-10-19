from pathlib import Path

from benchmark import execute
from benchmark_models import Config, Reference, Review
from metrics import proportion, summary
from models import Bias, Decision, Verdict
from provider import Status
import json


def test_every_metric_and_tie_denominators(tmp_path):
    config = Config.model_validate_json(Path('configs/demo.json').read_text())
    run = execute(config, tmp_path)
    for case in run.benchmark.cases:
        case.reference_review, case.reference_reviewer, case.ground_truth = Review.APPROVED, 'SYNTHETIC TEST', Reference.FIRST
        case.candidates[0].generator = config.judges[0]
        case.candidates[1].generator = config.judges[1]
        for t in case.transformations:
            t.review, t.reviewer, t.assertion = Review.APPROVED, 'SYNTHETIC TEST', 'fixture only'
    rows = {(r['judge'], r['metric']): r for r in summary(run)}
    for judge in config.judges:
        expected = {'position consistency':0, 'decisive flip rate':1, 'tie transition rate':0,
                    'repeatability':1, 'first-position preference':1, 'reviewed verbosity preference':.5,
                    'reviewed style preference':.5, 'own-answer preference':.5, 'reference agreement':.5,
                    'tie rate':0, 'invalid output rate':0, 'API failure rate':0}
        for metric, value in expected.items():
            assert rows[judge,metric]['value'] == value
            assert rows[judge,metric]['n'] > 0
    # One tie paired with a decisive verdict is a tie transition, not a decisive flip.
    base = [o for o in run.observations if o.bias == Bias.POSITION and o.judge == config.judges[0]]
    base[0].decision = Decision(verdict=Verdict.TIE, reason='fixture')
    rows = {r['metric']:r for r in summary(run) if r['judge']==config.judges[0]}
    assert rows['decisive flip rate']['n'] == 5
    assert rows['tie transition rate']['numerator'] == 1
    base[1].status, base[1].decision = Status.API_ERROR, None
    rows = {r['metric']:r for r in summary(run) if r['judge']==config.judges[0]}
    assert rows['position consistency']['n'] == 5
    assert rows['API failure rate']['numerator'] == 1


def test_unreviewed_excluded_and_cluster_interval(tmp_path):
    run=execute(Config.model_validate_json(Path('configs/demo.json').read_text()),tmp_path)
    for row in summary(run):
        if row['metric'] in ('reviewed verbosity preference','reviewed style preference','reference agreement','own-answer preference'):
            assert row['n']==0 and row['value'] is None
    samples=[('a',0),('a',0),('b',1),('b',1)]
    assert proportion(samples)==proportion(samples)
    assert proportion(samples)['ci95']==[0,1]
    assert proportion([('a',1)])['ci95'] is None
    assert proportion([])['value'] is None


def test_saved_software_regression_fixture(tmp_path):
    run=execute(Config.model_validate_json(Path('configs/demo.json').read_text()),tmp_path)
    expected=json.loads(Path('tests/expected-demo-metrics.json').read_text())
    assert summary(run)==expected
