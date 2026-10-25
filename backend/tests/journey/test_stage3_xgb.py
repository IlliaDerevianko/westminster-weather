"""Stage 3 XGBoost serving rules that need no database."""

from datetime import date

import pandas as pd
from sqlalchemy.exc import ProgrammingError

from westminster_weather.journey import stage3_xgb as s3

EVENTS = pd.DataFrame({"event_date": pd.to_datetime(["2026-10-01"])})


def _act(orders: int, appoint: float, ra: str) -> pd.Series:
    return pd.Series(
        {
            "commencement_orders_count": orders,
            "c_appoint": appoint,
            "royal_assent_date": pd.Timestamp(ra),
        }
    )


def test_recent_act_left_to_regulations_awaits_its_first_order():
    assert s3._awaiting_first_order(_act(0, 1.0, "2026-04-29"), EVENTS)


def test_act_with_an_order_or_no_regulation_clause_is_not_awaiting():
    assert not s3._awaiting_first_order(_act(2, 1.0, "2026-04-29"), EVENTS)
    assert not s3._awaiting_first_order(_act(0, 0.0, "2026-04-29"), EVENTS)


def test_old_act_without_orders_is_not_awaiting():
    assert not s3._awaiting_first_order(_act(0, 1.0, "2024-01-10"), EVENTS)


def test_missing_source_table_gives_no_forecast_instead_of_an_error(monkeypatch):
    def missing():
        raise ProgrammingError("select ...", {}, Exception('relation "bill_texts" does not exist'))

    monkeypatch.setattr(s3, "_DATA", None)
    monkeypatch.setattr(s3.cx, "load_tables", missing)
    assert s3.forecast(3137, date(2023, 10, 26)) is None
    assert s3._DATA is None  # not cached: retried once the table is back
