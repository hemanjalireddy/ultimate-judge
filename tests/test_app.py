import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('path', ['app.py', 'pages/1_Benchmark.py', 'pages/2_Dashboard.py'])
def test_streamlit_startup(path):
    app=AppTest.from_file(Path(path).resolve(),default_timeout=30).run()
    assert not app.exception


def test_missing_example_does_not_block_upload(monkeypatch):
    original = Path.read_bytes
    def read(path):
        if path.name == '30_questions.txt':
            raise FileNotFoundError('fixture: missing optional sample')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', read)
    app = AppTest.from_file(Path('app.py').resolve(), default_timeout=30).run()
    assert not app.exception
    assert any('still upload' in caption.value for caption in app.caption)
