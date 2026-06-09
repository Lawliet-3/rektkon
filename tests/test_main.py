from fastapi.testclient import TestClient

from app.main import app, model_service


class StubModel:
    def __init__(self, value=0.123):
        self.value = value

    def predict(self, frame):
        return [self.value]


def test_realtime_prediction_uses_model_when_available():
    model_service.production_model = StubModel(0.123)
    model_service.candidate_model = StubModel(0.124)
    client = TestClient(app)

    response = client.post(
        "/predict/realtime",
        json={"symbol": "BTCUSD", "recent_prices": [100.0, 101.0, 102.0], "volume": 10.0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_fallback"] is False
    assert body["prediction"] == 0.123


def test_realtime_prediction_falls_back_on_failure():
    model_service.production_model = None
    model_service.candidate_model = None
    client = TestClient(app)

    response = client.post(
        "/predict/realtime",
        json={"symbol": "EURUSD", "recent_prices": [1.1, 1.2, 1.18], "volume": 0.0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["used_fallback"] is True
    assert body["prediction"] > 0


def test_realtime_prediction_rejects_non_positive_price():
    client = TestClient(app)

    response = client.post(
        "/predict/realtime",
        json={"symbol": "BTCUSD", "recent_prices": [100.0, 0.0], "volume": 1.0},
    )

    assert response.status_code == 422


def test_realtime_prediction_rejects_negative_volume():
    client = TestClient(app)

    response = client.post(
        "/predict/realtime",
        json={"symbol": "BTCUSD", "recent_prices": [100.0, 101.0], "volume": -1.0},
    )

    assert response.status_code == 422
