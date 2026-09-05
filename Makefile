PY := .venv/bin/python

.PHONY: setup test eval optimize ask train-router

setup:
	python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt

test:
	$(PY) -m pytest -q

eval:
	$(PY) -m sqlsmith eval --lm $(or $(LM),fake)

optimize:
	$(PY) -m sqlsmith optimize --lm $(or $(LM),fake)

train-router:
	$(PY) -m sqlsmith train-router

ask:
	$(PY) -m sqlsmith ask "$(Q)" --lm $(or $(LM),fake)
