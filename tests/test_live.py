import os

import pytest

from provider import make_client, parse_decision, request, verify_models
from runner import MODELS


@pytest.mark.skipif(os.getenv('RUN_LIVE_TESTS') != '1' or not os.getenv('GROQ_API_KEY'), reason='Opt-in live test requires credentials')
def test_live_models_and_json():
    assert verify_models(MODELS)==sorted(MODELS)
    for model in MODELS:
        response=request(make_client(), model, 'Return JSON with verdict A and reason test.',{},json_mode=True)
        assert parse_decision(response.text).reason
