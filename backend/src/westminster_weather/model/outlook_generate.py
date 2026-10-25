"""Generate `bill_outlooks` for every Government Bill in Parliament on a date (default today).

Run from ``backend/``:
uv run python -m westminster_weather.model.outlook_generate [YYYY-MM-DD]
"""

from __future__ import annotations

import sys
from datetime import date

from westminster_weather.model import states
from westminster_weather.model.forecast import ForecastUnavailable
from westminster_weather.model.outlook import NotInParliament, build_outlook


def generate(as_of: date) -> list[dict]:
    bills = states._source(None).get_bills(as_of)
    live = bills[bills["outcome"] == "in_progress"]["bill_id"].astype(int).tolist()
    rows = []
    for bill_id in live:
        try:
            o = build_outlook(bill_id, as_of)
        except (ForecastUnavailable, NotInParliament) as exc:
            print(f"  {bill_id}: skipped ({exc})")
            continue
        rows.append(
            {
                "bill_id": o.bill_id,
                "forecast_date": o.forecast_date,
                "target_date": o.weather.target_date,
                "model_version": o.model_version,
                "weather_status": o.weather.status,
                "p_royal_assent": o.distribution.p_royal_assent,
                "p_by_target": o.weather.p_by_target,
                "recalibrated": o.recalibrated,
                "outlook": o.model_dump(mode="json"),
                "generated_at": o.generated_at,
            }
        )
        print(f"  {bill_id} {o.short_title}: {o.weather.headline}")
    return rows


def main() -> None:
    from westminster_weather.db import save_bill_outlooks_to_db

    as_of = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    rows = generate(as_of)
    print(f"saved {save_bill_outlooks_to_db(rows)} outlooks for {as_of} to bill_outlooks")


if __name__ == "__main__":
    main()
