import json
import logging
import os
import queue
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, make_asgi_app

logger = logging.getLogger("volatility_api")
logging.basicConfig(level=logging.INFO)

PRODUCTION_PREDICTION_HISTOGRAM = Histogram(
    "production_model_prediction_value", "Production model predicted volatility values"
)
CANDIDATE_PREDICTION_HISTOGRAM = Histogram(
    "candidate_model_prediction_value", "Candidate model predicted volatility values"
)
SHADOW_PREDICTION_DIFF_HISTOGRAM = Histogram(
    "shadow_prediction_abs_diff", "Absolute prediction difference between production and candidate models"
)
INFERENCE_LATENCY_HISTOGRAM = Histogram(
    "inference_latency_seconds", "Inference latency in seconds"
)
FALLBACK_TRIGGER_COUNTER = Counter(
    "fallback_trigger_total", "Number of times fallback prediction is used"
)
CANDIDATE_FAILURE_COUNTER = Counter(
    "candidate_prediction_failure_total", "Number of candidate prediction failures"
)
INFERENCE_PAYLOAD_LOG_COUNTER = Counter(
    "inference_payload_log_total", "Number of inference payloads written to storage"
)

INFERENCE_TIMEOUT_SECONDS = 0.5
DEFAULT_MOVING_AVERAGE_VOLATILITY = 0.02


class RealtimeRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol (e.g., BTCUSD)")
    recent_prices: list[float] = Field(..., min_length=1)
    volume: float = Field(0.0, ge=0, description="Recent traded volume")

    @field_validator("recent_prices")
    @classmethod
    def validate_recent_prices(cls, values: list[float]) -> list[float]:
        if any(value <= 0 for value in values):
            raise ValueError("All recent prices must be greater than zero")
        return values


class BatchRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    daily_closes: list[float] = Field(..., min_length=1)

    @field_validator("daily_closes")
    @classmethod
    def validate_daily_closes(cls, values: list[float]) -> list[float]:
        if any(value <= 0 for value in values):
            raise ValueError("All close prices must be greater than zero")
        return values


class PredictionResponse(BaseModel):
    symbol: str
    prediction: float
    confidence_interval: list[float]
    used_fallback: bool


class InferencePayloadLogger:
    def __init__(self) -> None:
        self.db_path = Path(os.getenv("INFERENCE_LOG_DB_PATH", "data/inference_payloads.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=10000)
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=1)

    def log(self, symbol: str, prices: list[float], volume: float) -> None:
        frame = ModelService._feature_frame(prices)
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "recent_prices": json.dumps(prices),
            "volume": float(volume),
            "last_price": float(frame["last_price"].iloc[0]),
            "price_mean": float(frame["price_mean"].iloc[0]),
            "price_std": float(frame["price_std"].iloc[0]),
        }
        try:
            self._queue.put_nowait(payload)
        except queue.Full:
            logger.warning("Inference payload log queue full; dropping payload")

    def _worker(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_payloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                symbol TEXT NOT NULL,
                recent_prices TEXT NOT NULL,
                volume REAL NOT NULL,
                last_price REAL NOT NULL,
                price_mean REAL NOT NULL,
                price_std REAL NOT NULL
            )
            """
        )
        conn.commit()

        while True:
            item = self._queue.get()
            if item is None:
                break
            conn.execute(
                """
                INSERT INTO inference_payloads
                (ts, symbol, recent_prices, volume, last_price, price_mean, price_std)
                VALUES (:ts, :symbol, :recent_prices, :volume, :last_price, :price_mean, :price_std)
                """,
                item,
            )
            conn.commit()
            INFERENCE_PAYLOAD_LOG_COUNTER.inc()

        conn.close()


class ModelService:
    def __init__(self) -> None:
        self.production_model: Any | None = None
        self.candidate_model: Any | None = None
        self.model_name = os.getenv("MLFLOW_MODEL_NAME", "volatility-model")

    def load_models(self) -> None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        self.production_model = self._load_model_by_stage("Production")
        self.candidate_model = self._load_model_by_stage("Candidate")

    def _load_model_by_stage(self, stage: str) -> Any | None:
        model_uri = f"models:/{self.model_name}/{stage}"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info("Loaded %s model from %s", stage.lower(), model_uri)
            return model
        except Exception as exc:
            logger.warning("Failed to load %s model %s: %s", stage.lower(), model_uri, exc)
            return None

    @staticmethod
    def _feature_frame(prices: list[float]) -> pd.DataFrame:
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

    def _predict_with_timeout(self, model: Any, prices: list[float]) -> float:
        frame = self._feature_frame(prices)

        def _model_predict() -> float:
            prediction = model.predict(frame)
            return float(np.asarray(prediction).reshape(-1)[0])

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_model_predict)
            return future.result(timeout=INFERENCE_TIMEOUT_SECONDS)

    def predict_production_with_fallback(self, prices: list[float]) -> tuple[float, bool]:
        if self.production_model is None:
            FALLBACK_TRIGGER_COUNTER.inc()
            return self._moving_average_fallback(prices), True

        try:
            prediction = self._predict_with_timeout(self.production_model, prices)
            PRODUCTION_PREDICTION_HISTOGRAM.observe(prediction)
            return prediction, False
        except (FutureTimeoutError, Exception):
            FALLBACK_TRIGGER_COUNTER.inc()
            prediction = self._moving_average_fallback(prices)
            PRODUCTION_PREDICTION_HISTOGRAM.observe(prediction)
            return prediction, True

    def predict_candidate(self, prices: list[float]) -> float | None:
        if self.candidate_model is None:
            CANDIDATE_FAILURE_COUNTER.inc()
            return None
        try:
            prediction = self._predict_with_timeout(self.candidate_model, prices)
            CANDIDATE_PREDICTION_HISTOGRAM.observe(prediction)
            return prediction
        except (FutureTimeoutError, Exception):
            CANDIDATE_FAILURE_COUNTER.inc()
            return None


app = FastAPI(title="Trading Signal & Volatility API", version="0.1.0")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
model_service = ModelService()
inference_payload_logger = InferencePayloadLogger()


@app.on_event("startup")
def startup() -> None:
    model_service.load_models()
    inference_payload_logger.start()


@app.on_event("shutdown")
def shutdown() -> None:
    inference_payload_logger.stop()


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
    start = time.perf_counter()
    production_prediction, used_fallback = model_service.predict_production_with_fallback(
        payload.recent_prices
    )
    candidate_prediction = model_service.predict_candidate(payload.recent_prices)

    if candidate_prediction is not None:
        SHADOW_PREDICTION_DIFF_HISTOGRAM.observe(
            abs(production_prediction - candidate_prediction)
        )

    latency = time.perf_counter() - start
    INFERENCE_LATENCY_HISTOGRAM.observe(latency)

    inference_payload_logger.log(payload.symbol, payload.recent_prices, payload.volume)

    ci_width = max(production_prediction * 0.1, 0.001)
    response = PredictionResponse(
        symbol=payload.symbol,
        prediction=production_prediction,
        confidence_interval=[production_prediction - ci_width, production_prediction + ci_width],
        used_fallback=used_fallback,
    )
    logger.info(
        "symbol=%s production_prediction=%f candidate_prediction=%s ci=%s fallback=%s",
        payload.symbol,
        response.prediction,
        candidate_prediction,
        response.confidence_interval,
        response.used_fallback,
    )
    return response


@app.post("/predict/batch")
def predict_batch(payload: BatchRequest) -> dict[str, Any]:
    batch_predictions = []
    for close_price in payload.daily_closes:
        pred, used_fallback = model_service.predict_production_with_fallback([close_price])
        batch_predictions.append({"prediction": pred, "used_fallback": used_fallback})
    return {
        "symbol": payload.symbol,
        "predictions": batch_predictions,
        "total": len(batch_predictions),
    }
