PY := .venv/bin/python

.PHONY: setup test eval optimize ask train-router publish-router serve bench parity

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

publish-router:
	$(PY) -m sqlsmith publish-router --backend $(or $(BACKEND),sklearn)

serve:
	$(PY) -m sqlsmith serve --lm $(or $(LM),fake) --router $(or $(ROUTER),sklearn) --port $(or $(PORT),8000)

bench:
	$(PY) -m sqlsmith bench --concurrency $(or $(C),16) --requests $(or $(N),200)

parity:
	$(PY) -m sqlsmith parity --lm $(or $(LM),fake)
