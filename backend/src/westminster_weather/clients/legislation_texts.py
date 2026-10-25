"""Ingests Act texts and in-force status from legislation.gov.uk into Neon.

For every Act in `act_commencements` (UK Government Bills that received Royal Assent):
1. `ukpga/{year}/{number}/enacted/data.xml`: the Act as enacted -> plain text in `act_texts`
   (read by the stage 3 model: how the Act says it comes into force).
2. `ukpga/{year}/{number}/data.xml`: the current revised version -> `act_in_force_status`:
   provisions still marked `Status="Prospective"` are not in force yet, so an Act is fully in
   force only when none are left; `latest_in_force_start_date` is the latest start date
   (`RestrictStartDate`) of any provision.

Downloads are cached under `backend/data/cache/legislation/` (gitignored) so a rerun resumes.
Run from `backend/`: uv run python -m westminster_weather.clients.legislation_texts
"""

from __future__ import annotations

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

log = logging.getLogger(__name__)

BASE_URL = "https://www.legislation.gov.uk/ukpga/{year}/{number}"
CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "legislation"
NS = "{http://www.legislation.gov.uk/namespaces/legislation}"
USER_AGENT = "westminster-weather/0.1 (legislation-ingestion)"
CONCURRENCY = 4
RETRIES = 4


async def _xml(client: httpx.AsyncClient, url: str, path: Path) -> bytes | None:
    if path.exists():
        return path.read_bytes()
    for attempt in range(RETRIES):
        try:
            r = await client.get(url)
        except httpx.TransportError:
            r = None
        if r is not None and r.status_code == 200 and r.content.lstrip().startswith(b"<"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(r.content)
            return r.content
        if r is not None and r.status_code == 404:
            return None
        await asyncio.sleep(2**attempt)
    log.warning("not fetched: %s", url)
    return None


def parse_enacted(xml: bytes) -> dict[str, Any]:
    root = ET.fromstring(xml)
    body = root.find(f"{NS}Primary")
    text = " ".join(" ".join((body if body is not None else root).itertext()).split())
    n = root.get("NumberOfProvisions")
    return {
        "text": text,
        "n_characters": len(text),
        "number_of_provisions": int(n) if n and n.isdigit() else None,
    }


def parse_current(xml: bytes) -> dict[str, Any]:
    root = ET.fromstring(xml)
    groups = list(root.iter(f"{NS}P1group"))
    prospective = [el for el in root.iter() if el.get("Status") == "Prospective"]
    starts = [d for el in root.iter() if (d := el.get("RestrictStartDate"))]
    latest = max((date.fromisoformat(d[:10]) for d in starts), default=None)
    return {
        "n_section_groups": len(groups),
        "n_section_groups_prospective": sum(g.get("Status") == "Prospective" for g in groups),
        "n_prospective_elements": len(prospective),
        "fully_in_force": not prospective,
        "latest_in_force_start_date": latest,
    }


async def _one(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, act: pd.Series, now: datetime
) -> tuple[dict | None, dict | None]:
    base = BASE_URL.format(year=int(act["ukpga_year"]), number=int(act["ukpga_number"]))
    key = f"{int(act['ukpga_year'])}_{int(act['ukpga_number'])}"
    async with sem:
        enacted = await _xml(client, f"{base}/enacted/data.xml", CACHE_DIR / f"{key}_enacted.xml")
        current = await _xml(client, f"{base}/data.xml", CACHE_DIR / f"{key}_current.xml")
    text_row = status_row = None
    try:
        if enacted:
            text_row = {
                "bill_id": int(act["bill_id"]),
                "ukpga_year": int(act["ukpga_year"]),
                "ukpga_number": int(act["ukpga_number"]),
                "act_title": act["act_title"],
                "source_url": f"{base}/enacted",
                "fetched_at": now,
                **parse_enacted(enacted),
            }
        if current:
            status_row = {
                "bill_id": int(act["bill_id"]),
                "source_url": base,
                "checked_at": now,
                **parse_current(current),
            }
    except ET.ParseError as exc:
        log.warning("%s: unreadable XML (%s)", key, exc)
    return text_row, status_row


async def fetch_all(acts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    now = datetime.now(UTC)
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(
        timeout=60.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        results = await asyncio.gather(*(_one(client, sem, a, now) for _, a in acts.iterrows()))
    texts = pd.DataFrame([t for t, _ in results if t])
    status = pd.DataFrame([s for _, s in results if s])
    return texts, status


def main() -> None:
    from sqlalchemy import text

    from westminster_weather.db import get_sync_pooled_engine, save_act_texts_to_db

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    e = get_sync_pooled_engine()
    try:
        acts = pd.read_sql(
            text(
                "select bill_id, act_title, ukpga_year, ukpga_number "
                "from act_commencements order by bill_id"
            ),
            e,
        )
    finally:
        e.dispose()
    print(f"{len(acts)} Acts")
    texts, status = asyncio.run(fetch_all(acts))
    print(f"texts: {len(texts)}  in-force status: {len(status)}")
    if len(status):
        print(f"fully in force: {int(status['fully_in_force'].sum())} of {len(status)}")
    print(save_act_texts_to_db(texts, status))


if __name__ == "__main__":
    main()
