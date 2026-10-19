from benchmark_models import Constraints, Facts
from dataset import SCENARIOS
from dataset import build
import csv
import json


def test_constraint_boundaries_and_missing_director():
    movie=Facts(id=1,title='fixture',genres=['Drama'],runtime=120,year=2015,rating=7,votes=1000)
    checks=Constraints(genre='Drama',max_runtime=120,min_year=2015,min_rating=7,min_votes=1000,director='Director').check(movie)
    assert checks == {'genre':True,'runtime':False,'year':True,'rating':True,'votes':True,'director':None}
    assert len(SCENARIOS)*5==30


def test_builder_produces_thirty_frozen_cases(tmp_path):
    path=tmp_path/'movies.csv'
    with path.open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=['id','title','genres','runtime','release_date','vote_average','vote_count'])
        writer.writeheader()
        for i in range(10):
            writer.writerow(dict(id=i,title=f'Fixture {i}',genres=json.dumps([{'name':g} for g in ['Science Fiction','Thriller','Animation','Comedy','Drama','Family']]),
                                 runtime=90 if i<5 else 150,release_date='2016-01-01' if i<5 else '2000-01-01',vote_average=8 if i<5 else 4,vote_count=2000 if i<5 else 10))
    benchmark=build(path)
    assert len(benchmark.cases)==30
    assert benchmark==build(path)
    for case in benchmark.cases:
        assert sum(all(v is True for v in case.constraints.check(m).values()) for m in case.movie_facts)==1
