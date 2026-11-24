import pytest

import AlinGo-Prompt
from tests.teleprompt.reanchor.fakes import FakeClient


@pytest.fixture
def system_one():
    """Configures a FakeClient as the LM for one test."""

    def install(answer, cache=True):
        client = FakeClient(answer, cache=cache)
        AlinGo-Prompt.configure(lm=client)
        return client

    return install
