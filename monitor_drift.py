import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from train import load_data


FEATURE_COLUMNS = ["last_price", "price_mean", "price_std"]


def load_inference_data(db_path: Path) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
    SELECT last_price, price_mean, price_std
    FROM inference_payloads
    ORDER BY id DESC
    LIMIT 5000
    """
    frame = pd.read_sql_query(query, conn)
    conn.close()
    return frame


def main() -> int:
    db_path = Path(os.getenv("INFERENCE_LOG_DB_PATH", "data/inference_payloads.db"))
    output_dir = Path(os.getenv("DRIFT_REPORT_DIR", "reports"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        print(f"No inference log found at {db_path}; skipping drift report")
        return 0

    baseline = load_data()[FEATURE_COLUMNS].copy()
    current = load_inference_data(db_path)

    if current.empty:
        print("No live inference rows found; skipping drift report")
        return 0

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline, current_data=current)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    html_path = output_dir / f"drift_report_{timestamp}.html"
    json_path = output_dir / f"drift_report_{timestamp}.json"

    report.save_html(str(html_path))
    json_path.write_text(report.json(), encoding="utf-8")

    print(f"Generated drift reports: {html_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
