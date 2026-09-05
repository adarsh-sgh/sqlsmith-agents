"""Pluggable DSPy language models: Claude via the Anthropic SDK, and a deterministic fake for tests."""

import re
import zlib
from types import SimpleNamespace

import dspy

from sqlsmith.gold import GOLD

DEFAULT_MODEL = "claude-sonnet-5"


def _response(text, model):
    """Minimal object satisfying dspy.BaseLM._process_lm_response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop")],
        usage={"prompt_tokens": 0, "completion_tokens": 0},
        model=model,
    )


class ClaudeLM(dspy.BaseLM):
    """dspy.BaseLM backed by anthropic.Anthropic().messages.create."""

    def __init__(self, model=DEFAULT_MODEL, max_tokens=2048, **kwargs):
        super().__init__(model=model, max_tokens=max_tokens, cache=False, **kwargs)
        import anthropic  # imported lazily so tests never need the package configured

        self.client = anthropic.Anthropic()

    def forward(self, prompt=None, messages=None, **kwargs):
        messages = messages or [{"role": "user", "content": prompt}]
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        extra = {"system": system} if system else {}
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.kwargs["max_tokens"], messages=chat, **extra
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _response(text, self.model)


_FIELD = re.compile(r"\[\[ ## (\w+) ## \]\]\n(.*?)(?=\n\n\[\[ ## |\Z)", re.S)
_OUT_FIELDS = re.compile(r"Your output fields are:\n(.*?)\n\n", re.S)


class FakeLM(dspy.BaseLM):
    """Deterministic oracle over the gold set that speaks DSPy's ChatAdapter format.

    Failure injection (stable per question via crc32):
      bucket 0-1: first SQL references a bogus column -> execution error -> repair fixes it.
      bucket 2:   SQL runs but returns the wrong result (uncatchable by the critic).
    This makes execution accuracy < 100% and the repair loop observable in tests and evals.
    """

    BUCKETS = 10

    def __init__(self, break_first=(0, 1), wrong=(2,)):
        super().__init__(model="fake-oracle", cache=False)
        self.answers = {g.question: g.sql for g in GOLD}
        self.break_first, self.wrong = set(break_first), set(wrong)
        self.calls = 0

    @staticmethod
    def bucket(question):
        return zlib.crc32(question.encode()) % FakeLM.BUCKETS

    def _first_sql(self, question):
        gold = self.answers.get(question, "SELECT COUNT(*) FROM customers")
        b = self.bucket(question)
        if b in self.break_first:
            return gold.replace("SELECT ", "SELECT no_such_col, ", 1)
        if b in self.wrong:
            return f"SELECT * FROM ({gold}) WHERE 0"
        return gold

    def _value(self, name, inputs):
        q = inputs.get("question", "")
        gold = self.answers.get(q, "SELECT COUNT(*) FROM customers")
        if name == "tables":
            known = re.findall(r"CREATE TABLE (\w+)", inputs.get("ddl", ""))
            return ", ".join(t for t in known if re.search(rf"\b{t}\b", gold)) or "customers"
        if name == "sql":
            return self._first_sql(q)
        if name == "fixed_sql":
            return gold
        if name == "reasoning":
            return f"fake reasoning for bucket {self.bucket(q)}"
        return "unknown"

    def forward(self, prompt=None, messages=None, **kwargs):
        self.calls += 1
        messages = messages or [{"role": "user", "content": prompt}]
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = messages[-1]["content"]
        inputs = {k: v.strip() for k, v in _FIELD.findall(user)}
        spec = _OUT_FIELDS.search(system)
        out_names = re.findall(r"`(\w+)`", spec.group(1)) if spec else ["sql"]
        body = "\n\n".join(f"[[ ## {n} ## ]]\n{self._value(n, inputs)}" for n in out_names)
        return _response(body + "\n\n[[ ## completed ## ]]", self.model)


def make_lm(name, model=DEFAULT_MODEL):
    if name == "fake":
        return FakeLM()
    if name == "claude":
        return ClaudeLM(model=model)
    raise ValueError(f"unknown lm '{name}' (use fake|claude)")
