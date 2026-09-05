import dspy
import pytest

from sqlsmith.db import build_db
from sqlsmith.lm import FakeLM


@pytest.fixture(scope="session")
def conn():
    return build_db()


@pytest.fixture
def fake_lm():
    lm = FakeLM()
    dspy.configure(lm=lm)
    return lm
