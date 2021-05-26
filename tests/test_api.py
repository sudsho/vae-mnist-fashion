"""FastAPI smoke tests using TestClient.

These tests stub out the model loader so we don't need a real checkpoint.
"""
import torch
import pytest

from src.api import main as api_main
from src.model import VAE


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    fake = VAE(latent_dim=8)
    fake.eval()
    monkeypatch.setattr(api_main, "_load", lambda: fake)
    api_main._MODEL = None
    yield


def test_health():
    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_generate():
    from fastapi.testclient import TestClient
    client = TestClient(api_main.app)
    r = client.post("/generate", json={"n": 3})
    assert r.status_code == 200
    assert len(r.json()["images"]) == 3
