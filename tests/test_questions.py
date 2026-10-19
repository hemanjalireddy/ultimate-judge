import json

import pytest

from questions import load_catalog, make_scenarios, parse_questions
from questions import automatic_movie_ids


def catalog_bytes():
    import csv
    import io
    text=io.StringIO()
    writer=csv.DictWriter(text,fieldnames=['id','title','genres','runtime','release_date','vote_average','vote_count'])
    writer.writeheader()
    for i in (1,2,3):
        writer.writerow(dict(id=i,title=f'Film {i}',genres=json.dumps([{'name':'Drama'}]),runtime=90+i,
                             release_date='2020-01-01',vote_average=7,vote_count=100))
    return text.getvalue().encode()


def test_text_questions_have_no_answers_and_use_shared_facts():
    questions=parse_questions(b'Pick the shortest movie.\n\nPick the highest rated movie.','questions.txt')
    scenarios=make_scenarios(questions,catalog_bytes(),[1,2])
    assert len(scenarios.cases)==2
    assert 'candidates' not in scenarios.model_dump_json()
    assert scenarios.cases[0].movie_facts[0].title=='Film 1'


def test_json_ids_override_shared_options():
    questions=parse_questions(json.dumps({'questions':[{'prompt':'Recommend a movie','movie_ids':[2,3]}]}).encode(),'q.json')
    scenarios=make_scenarios(questions,catalog_bytes(),[1,2])
    assert [m.id for m in scenarios.cases[0].movie_facts]==[2,3]
    assert set(load_catalog(catalog_bytes()))=={1,2,3}


@pytest.mark.parametrize('content', [b'[]',b'{}',b'[{"prompt":" "}]',b'[{"id":"x","prompt":"a"},{"id":"x","prompt":"b"}]'])
def test_bad_questions_rejected(content):
    with pytest.raises(ValueError):
        parse_questions(content,'q.json')


def test_missing_movie_id_is_not_invented():
    questions=parse_questions(b'[{"prompt":"Pick","movie_ids":[1,999]}]','q.json')
    with pytest.raises(ValueError,match='not found'):
        make_scenarios(questions,catalog_bytes(),[])


def test_automatic_fact_selection_is_stable():
    questions=parse_questions(b'Recommend a drama.','q.txt')
    catalog=load_catalog(catalog_bytes())
    ids=automatic_movie_ids(questions[0],catalog)
    assert ids==automatic_movie_ids(questions[0],catalog)==[1,2,3]
    scenarios=make_scenarios(questions,catalog_bytes())
    assert [m.id for m in scenarios.cases[0].movie_facts]==ids
