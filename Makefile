PY=python
PIP=pip

.PHONY: setup lint type test dq features ge-docs clean

setup:
	$(PIP) install -U pip
	$(PIP) install -e . -e .[dev]
	pre-commit install

lint:
	ruff check . && ruff format --check . && black --check .

type:
	mypy src/driftradar_vision

test:
	pytest

features:
	$(PY) -m driftradar_vision.features --config-path configs --config-name data.yaml

dq:
	$(PY) -m driftradar_vision.dq_validate --config-path configs --config-name dq.yaml

clean:
	rm -rf artifacts mlruns .pytest_cache .mypy_cache
