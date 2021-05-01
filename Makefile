PY=python

.PHONY: install train sample test clean

install:
	pip install -r requirements.txt

train:
	$(PY) -m src.train --config configs/default.yaml

sample:
	$(PY) -m src.sample --ckpt runs/default/last.pt

test:
	pytest -q

clean:
	rm -rf runs/ samples/ __pycache__
