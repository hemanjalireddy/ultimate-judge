from concurrent.futures import ThreadPoolExecutor

import pytest
from filelock import FileLock
from pydantic import BaseModel

from benchmark import generate_candidates, write_json
from provider import ProviderError


class Record(BaseModel):
    value: int


def test_concurrent_atomic_saves_have_separate_temp_files(tmp_path):
    path = tmp_path / 'checkpoint.json'
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: write_json(path, Record(value=i)), range(40)))
    assert 0 <= Record.model_validate_json(path.read_text()).value < 40
    assert not list(tmp_path.glob('*.tmp'))


def test_duplicate_generation_is_blocked_without_touching_answers(tmp_path):
    output = tmp_path / 'answers.json'
    with FileLock(str(output.with_suffix('.generation.lock'))):
        with pytest.raises(ProviderError, match='already being generated'):
            generate_candidates(tmp_path / 'questions.json', output)
    assert not output.exists()
