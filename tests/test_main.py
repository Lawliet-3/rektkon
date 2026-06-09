from fastapi.testclient import TestClient

from app.main import app, model_service


class StubModel:
    def predict(self, frame):
        return [0.123]


def test_realtime_prediction_uses_model_when_available():
    model_service.model = StubModel()
    client = TestClient(app)

    response = client.post(
        "/predict/realtime",
        json={"symbol": "BTCUSD", "recent_prices": [100.0, 101.0, 102.0]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_fallback"] is False
    assert body["prediction"] == 0.123


def test_realtime_prediction_falls_back_on_failure():
    model_service.model = None
    client = TestClient(app)

    response = client.post(
        "/predict/realtime",
        json={"symbol": "EURUSD", "recent_prices": [1.1, 1.2, 1.18]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_fallback"] is True
    assert body["prediction"] > 0
