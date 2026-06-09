import os
import sqlite3
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class LiveFeatureRecord(BaseModel):
    last_price: float = Field(..., gt=0)
    price_mean: float = Field(..., gt=0)
    price_std: float = Field(..., ge=0)
    volume: float = Field(..., ge=0)


def main() -> int:
    db_path = Path(os.getenv("INFERENCE_LOG_DB_PATH", "data/inference_payloads.db"))
    if not db_path.exists():
        print(f"No inference log found at {db_path}; skipping validation")
        return 0

    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        """
        SELECT id, last_price, price_mean, price_std, volume
        FROM inference_payloads
        ORDER BY id DESC
        LIMIT 1000
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("No inference payload rows available; skipping validation")
        return 0

    invalid_rows: list[tuple[int, str]] = []
    for row in rows:
        row_id, last_price, price_mean, price_std, volume = row
        try:
            LiveFeatureRecord(
                last_price=last_price,
                price_mean=price_mean,
                price_std=price_std,
                volume=volume,
            )
        except ValidationError as exc:
            invalid_rows.append((int(row_id), str(exc)))

    if invalid_rows:
        print(f"Validation failed for {len(invalid_rows)} rows")
        for row_id, err in invalid_rows[:20]:
            print(f"row_id={row_id}: {err}")
        return 1

    print(f"Validation succeeded for {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
