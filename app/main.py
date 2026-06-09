import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

logger = logging.getLogger("volatility_api")
logging.basicConfig(level=logging.INFO)

PREDICTION_HISTOGRAM = Histogram(
    "model_prediction_value", "Predicted volatility values"
)
INFERENCE_LATENCY_HISTOGRAM = Histogram(
    "inference_latency_seconds", "Inference latency in seconds"
)
FALLBACK_TRIGGER_COUNTER = Counter(
    "fallback_trigger_total", "Number of times fallback prediction is used"
)

INFERENCE_TIMEOUT_SECONDS = 0.5
DEFAULT_MOVING_AVERAGE_VOLATILITY = 0.02


class RealtimeRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g., BTCUSD)")
    recent_prices: list[float] = Field(..., min_length=1)


class BatchRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    daily_closes: list[float] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    confidence_interval: list[float]
    used_fallback: bool


class ModelService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.model_name = os.getenv("MLFLOW_MODEL_NAME", "volatility-model")

    def _load_production_model(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{self.model_name}/Production"
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Loaded production model from %s", model_uri)
        except Exception as exc:  # fallback is expected when model not ready
            self.model = None
            logger.warning("Failed to load model %s: %s", model_uri, exc)

    def _feature_frame(self, prices: list[float]) -> pd.DataFrame:
        arr = np.asarray(prices, dtype=float)
        return pd.DataFrame(
            {
                "last_price": [float(arr[-1])],
                "price_mean": [float(arr.mean())],
                "price_std": [float(arr.std())],
            }
        )

    def _moving_average_fallback(self, prices: list[float]) -> float:
        arr = np.asarray(prices, dtype=float)
        if len(arr) < 2:
            return DEFAULT_MOVING_AVERAGE_VOLATILITY
        returns = np.diff(arr) / np.clip(arr[:-1], a_min=1e-12, a_max=None)
        return float(np.mean(np.abs(returns)))

    def predict_with_fallback(self, prices: list[float]) -> tuple[float, bool]:
        start = time.perf_counter()
        used_fallback = False

        def _model_predict() -> float:
            if self.model is None:
                raise RuntimeError("Production model unavailable")
            frame = self._feature_frame(prices)
            prediction = self.model.predict(frame)
            return float(np.asarray(prediction).reshape(-1)[0])

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_model_predict)
                prediction = future.result(timeout=INFERENCE_TIMEOUT_SECONDS)
        except (FutureTimeoutError, Exception):
            used_fallback = True
            FALLBACK_TRIGGER_COUNTER.inc()
            prediction = self._moving_average_fallback(prices)

        latency = time.perf_counter() - start
        INFERENCE_LATENCY_HISTOGRAM.observe(latency)
        PREDICTION_HISTOGRAM.observe(prediction)
        return prediction, used_fallback


app = FastAPI(title="Trading Signal & Volatility API", version="0.1.0")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
model_service = ModelService()


@app.on_event("startup")
def load_production_model_on_startup() -> None:
    model_service._load_production_model()


@app.middleware("http")
async def log_request_metrics(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info(
        "path=%s method=%s status=%s latency_ms=%.2f",
        request.url.path,
        request.method,
        response.status_code,
        duration * 1000,
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/realtime", response_model=PredictionResponse)
def predict_realtime(payload: RealtimeRequest) -> PredictionResponse:
    prediction, used_fallback = model_service.predict_with_fallback(payload.recent_prices)
    ci_width = max(prediction * 0.1, 0.001)
    response = PredictionResponse(
        symbol=payload.symbol,
        prediction=prediction,
        confidence_interval=[prediction - ci_width, prediction + ci_width],
        used_fallback=used_fallback,
    )
    logger.info(
        "symbol=%s prediction=%f ci=%s fallback=%s",
        payload.symbol,
        response.prediction,
        response.confidence_interval,
        response.used_fallback,
    )
    return response


@app.post("/predict/batch")
def predict_batch(payload: BatchRequest) -> dict[str, Any]:
    batch_predictions = []
    for close_price in payload.daily_closes:
        pred, used_fallback = model_service.predict_with_fallback([close_price])
        batch_predictions.append({"prediction": pred, "used_fallback": used_fallback})
    return {
        "symbol": payload.symbol,
        "predictions": batch_predictions,
        "total": len(batch_predictions),
    }
