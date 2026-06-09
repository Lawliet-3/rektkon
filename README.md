# rektkon

Production-ready ML microservice scaffold for trading signal and volatility forecasting.

## Standard project layout

- `app/main.py` - FastAPI inference service (`/health`, `/predict/realtime`, `/predict/batch`, `/metrics`)
- `train.py` - model training + MLflow logging and model registration
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - API + MLflow + Prometheus + Grafana stack
- `monitoring/prometheus.yml` - Prometheus scrape configuration
- `tests/test_main.py` - focused API tests
