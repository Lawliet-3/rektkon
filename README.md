# rektkon

Production-ready ML microservice scaffold for trading signal and volatility forecasting.

## Standard project layout

- `app/main.py` - FastAPI inference service (`/health`, `/predict/realtime`, `/predict/batch`, `/metrics`) with shadow deployment (`Production` + `Candidate`) and inference payload logging to SQLite.
- `train.py` - model training + MLflow logging and model registration
- `validate_data.py` - validates logged live feature data with Pydantic checks
- `monitor_drift.py` - generates Evidently data drift reports from live inference logs vs training baseline
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - API + MLflow + monitoring worker + Prometheus + Grafana stack
- `monitoring/prometheus.yml` - Prometheus scrape configuration
- `tests/test_main.py` - focused API tests
