PY=python

.PHONY: install train train-beta train-cnn train-cvae sample app api test clean

install:
	pip install -r requirements.txt

train:
	$(PY) -m src.train --config configs/default.yaml

train-beta:
	$(PY) -m src.train --config configs/beta_vae.yaml

train-cnn:
	$(PY) -m src.train --config configs/cnn.yaml

train-cvae:
	$(PY) -m src.train_cvae --config configs/cvae.yaml

sample:
	$(PY) -m src.sample --ckpt runs/default/last.pt

app:
	streamlit run streamlit_app.py

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	pytest -q

clean:
	rm -rf runs/ samples/ __pycache__
