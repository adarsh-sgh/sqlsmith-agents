"""ClaudeLM against a stubbed Anthropic client: no key, no network."""

from types import SimpleNamespace

import dspy
import pytest

from sqlsmith import lm as lm_mod
from sqlsmith.agents import WriteSQL


class _Client:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=self.text)])


def test_claude_lm_round_trips_through_dspy_predict(monkeypatch):
    client = _Client("[[ ## sql ## ]]\nSELECT COUNT(*) FROM customers\n\n[[ ## completed ## ]]")
    monkeypatch.setattr("anthropic.Anthropic", lambda: client)
    lm = lm_mod.ClaudeLM()
    dspy.configure(lm=lm)
    assert dspy.Predict(WriteSQL)(question="How many customers?", ddl="CREATE TABLE customers (id)").sql == "SELECT COUNT(*) FROM customers"
    (call,) = client.calls
    assert call["model"] == lm_mod.DEFAULT_MODEL and call["system"] and call["messages"][-1]["role"] == "user"


def test_claude_lm_surfaces_refusal(monkeypatch):
    monkeypatch.setattr("anthropic.Anthropic", lambda: _Client("", stop_reason="refusal"))
    with pytest.raises(RuntimeError, match="refused"):
        lm_mod.ClaudeLM().forward(messages=[{"role": "user", "content": "hi"}])
