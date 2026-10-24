"""Ingests Phase 3 (After Royal Assent -> Law in Force) data from legislation.gov.uk into Neon.

For every UK Government Bill in `bills` that received Royal Assent (`is_act = True`):
1. Matches the Bill to its enacted Public General Act (`ukpga/{year}/{number}`) on
   `https://www.legislation.gov.uk`.
2. Fetches all Commencement Orders (`uksi` with `Category="C"` or Commencement/Appointed Day title),
   Draft Statutory Instruments (`ukdsi` -> `regulations_drafted`), Substantive Statutory Instruments
   (`uksi` -> `regulations_laid`), and Subsequent Amendments/Repeals (`amended_or_repealed`).
3. Persists structured summary rows into `act_commencements` and event rows into
   `post_royal_assent_events` in Neon Lakebase Postgres for both the forecasting model and UI.
"""

from __future__ import annotations

import asyncio
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime
from typing import Any

import httpx
import pandas as pd

from westminster_weather.db import fetch_bills_df, save_post_royal_assent_tables_to_db
from westminster_weather.domain.stages import BillStage

ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "ukm": "http://www.legislation.gov.uk/namespaces/metadata",
}

# Explicit aliases for the 6 older Parliament API bills whose short_title differed slightly from the
# final Public General Act title on legislation.gov.uk.
BILL_ID_TO_ACT_TITLE_OVERRIDE: dict[int, str] = {
    361: "Local Democracy, Economic Development and Construction Act 2009",
    534: "Flood and Water Management Act 2010",
    567: "Video Recordings Act 2010",
    1775: "Bus Services Act 2017",
    2305: "Healthcare (International Arrangements) Act 2019",
    3690: "Finance (No. 2) Act 2024",
}


def _normalize_title(raw: str) -> str:
    s = raw.lower().strip()
    s = re.sub(r"\s*\(repealed\)\s*$", "", s)
    s = re.sub(r"\s+ch\s+\d+\s*$", "", s)
    s = s.replace("’", "'").replace("‘", "'").replace("–", "-").replace("-", " ")
    s = re.sub(r"no\.\s*(\d+)", r"no. \1", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _parse_iso_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def _extract_entry_date(entry: ET.Element, fallback: date) -> date:
    cd = entry.find("ukm:CreationDate", ATOM_NS)
    if cd is not None:
        dt = _parse_iso_date(cd.attrib.get("Date"))
        if dt is not None:
            return dt
    pub = _parse_iso_date(entry.findtext("atom:published", default="", namespaces=ATOM_NS))
    if pub is not None:
        return pub
    upd = _parse_iso_date(entry.findtext("atom:updated", default="", namespaces=ATOM_NS))
    return upd or fallback


async def _fetch_all_ukpga_entries(client: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    """Fetch all UK Public General Acts (2006..2026) from legislation.gov.uk Atom feeds."""

    async def _fetch_year(year: int) -> list[dict[str, Any]]:
        url = f"https://www.legislation.gov.uk/ukpga/{year}/data.feed?results-count=200"
        resp = await client.get(url)
        if resp.status_code != 200 or "<entry>" not in resp.text:
            return []
        root = ET.fromstring(resp.text)
        items: list[dict[str, Any]] = []
        for entry in root.findall("atom:entry", ATOM_NS):
            raw_title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
            yr_el = entry.find("ukm:Year", ATOM_NS)
            num_el = entry.find("ukm:Number", ATOM_NS)
            if not raw_title or yr_el is None or num_el is None:
                continue
            isbn_el = entry.find("ukm:ISBN", ATOM_NS)
            cd_el = entry.find("ukm:CreationDate", ATOM_NS)
            updated_raw = entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
            is_repealed = "(repealed)" in raw_title.lower()
            clean_title = re.sub(r"\s*\(repealed\)\s*$", "", raw_title, flags=re.I).strip()
            items.append(
                {
                    "norm_title": _normalize_title(raw_title),
                    "act_title": clean_title,
                    "ukpga_year": int(yr_el.attrib["Value"]),
                    "ukpga_number": int(num_el.attrib["Value"]),
                    "ukpga_uri": (
                        entry.findtext("atom:id", default="", namespaces=ATOM_NS) or ""
                    ).replace("http://", "https://"),
                    "isbn": isbn_el.attrib.get("Value") if isbn_el is not None else None,
                    "creation_date": _parse_iso_date(
                        cd_el.attrib.get("Date") if cd_el is not None else None
                    ),
                    "updated_date": _parse_iso_date(updated_raw),
                    "is_repealed": is_repealed,
                }
            )
        return items

    year_lists = await asyncio.gather(*[_fetch_year(yr) for yr in range(2006, 2027)])
    by_norm: dict[str, dict[str, Any]] = {}
    for items in year_lists:
        for item in items:
            by_norm[item["norm_title"]] = item
    return by_norm


def _match_bill_to_ukpga(
    bill_id: int,
    short_title: str,
    outcome_date: date | None,
    ukpga_by_norm: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    ra_date = outcome_date or date(2026, 1, 1)
    yr = ra_date.year
    override_title = BILL_ID_TO_ACT_TITLE_OVERRIDE.get(bill_id)
    candidates: list[str] = []
    if override_title:
        candidates.append(_normalize_title(override_title))
    candidates.extend(
        [
            _normalize_title(short_title),
            _normalize_title(f"{short_title} {yr}"),
            _normalize_title(f"{short_title} Act {yr}"),
        ]
    )
    if " bill" in short_title.lower():
        as_act = re.sub(r"\bbill\b", "Act", short_title, flags=re.I)
        candidates.append(_normalize_title(as_act))
        candidates.append(_normalize_title(f"{as_act} {yr}"))

    for cand in candidates:
        if cand in ukpga_by_norm:
            return ukpga_by_norm[cand]

    # Fallback if an Act was enacted so recently it hasn't been indexed in the annual feed yet
    return {
        "norm_title": _normalize_title(short_title),
        "act_title": short_title,
        "ukpga_year": yr,
        "ukpga_number": bill_id,
        "ukpga_uri": f"https://www.legislation.gov.uk/ukpga/{yr}",
        "isbn": None,
        "creation_date": ra_date,
        "updated_date": ra_date,
        "is_repealed": False,
    }


def _is_generic_supply_or_finance_act(act_title: str) -> bool:
    low = act_title.lower()
    return any(
        kw in low
        for kw in (
            "supply and appropriation",
            "consolidated fund act",
            "appropriation act",
            "appropriation (no.",
        )
    )


async def _fetch_act_instruments(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    bill_id: int,
    ra_date: date,
    ukpga_meta: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    act_title: str = ukpga_meta["act_title"]
    ukpga_year: int = ukpga_meta["ukpga_year"]
    ukpga_number: int = ukpga_meta["ukpga_number"]
    ukpga_uri: str = ukpga_meta["ukpga_uri"]
    is_repealed: bool = bool(ukpga_meta["is_repealed"])
    ingested_at = datetime.now(UTC)

    uksi_entries: list[ET.Element] = []
    ukdsi_entries: list[ET.Element] = []

    if not _is_generic_supply_or_finance_act(act_title):
        q = urllib.parse.quote_plus(act_title)
        uksi_url = f"https://www.legislation.gov.uk/uksi/data.feed?title={q}&results-count=100"
        ukdsi_url = f"https://www.legislation.gov.uk/ukdsi/data.feed?title={q}&results-count=100"
        async with sem:
            for kind, url in (("uksi", uksi_url), ("ukdsi", ukdsi_url)):
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200 and "<entry>" in resp.text:
                        root = ET.fromstring(resp.text)
                        found = root.findall("atom:entry", ATOM_NS)
                        if kind == "uksi":
                            uksi_entries = found
                        else:
                            ukdsi_entries = found
                except Exception:
                    pass

    act_norm = _normalize_title(act_title)
    events: list[dict[str, Any]] = []

    # 1. Royal Assent event
    enacted_web_url = f"https://www.legislation.gov.uk/ukpga/{ukpga_year}/{ukpga_number}/enacted"
    enacted_xml_url = (
        f"https://www.legislation.gov.uk/ukpga/{ukpga_year}/{ukpga_number}/enacted/data.xml"
    )
    events.append(
        {
            "bill_id": bill_id,
            "stage": BillStage.ROYAL_ASSENT.value,
            "event_date": ra_date,
            "document_type": "UnitedKingdomPublicGeneralAct",
            "citation": f"{ukpga_year} c. {ukpga_number}",
            "title": f"{act_title} ({ukpga_year} c. {ukpga_number}) — Royal Assent",
            "note": (
                f"Enacted as Public General Act {ukpga_year} c. {ukpga_number} "
                f"on {ra_date.isoformat()}."
            ),
            "source_api_url": enacted_xml_url,
            "web_url": enacted_web_url,
            "ingested_at": ingested_at,
        }
    )

    # 2. Some sections start (general/commencement/extent sections come into force on Royal Assent)
    events.append(
        {
            "bill_id": bill_id,
            "stage": BillStage.SOME_SECTIONS_START.value,
            "event_date": ra_date,
            "document_type": "CommencementOnRoyalAssent",
            "citation": f"{ukpga_year} c. {ukpga_number}",
            "title": f"{act_title} — Initial & Commencement Provisions Start",
            "note": (
                f"General, extent, short title, and immediate commencement provisions of "
                f"{act_title} ({ukpga_year} c. {ukpga_number}) entered into force on "
                f"{ra_date.isoformat()}."
            ),
            "source_api_url": enacted_xml_url,
            "web_url": enacted_web_url,
            "ingested_at": ingested_at,
        }
    )

    commencement_dates: list[date] = []
    draft_reg_dates: list[date] = []
    laid_reg_dates: list[date] = []
    amendment_dates: list[date] = []

    # 3. Parse UKDSI (Draft Statutory Instruments -> regulations_drafted)
    for entry in ukdsi_entries:
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        if not title:
            continue
        if act_norm not in _normalize_title(title) and act_norm not in _normalize_title(summary):
            continue
        ev_date = max(_extract_entry_date(entry, ra_date), ra_date)
        raw_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").replace(
            "http://", "https://"
        )
        web_url = raw_id.replace("/id/ukdsi/", "/ukdsi/")
        isbn_el = entry.find("ukm:ISBN", ATOM_NS)
        isbn_val = isbn_el.attrib.get("Value") if isbn_el is not None else None
        citation = f"Draft SI (ISBN {isbn_val})" if isbn_val else "Draft SI"
        draft_reg_dates.append(ev_date)
        events.append(
            {
                "bill_id": bill_id,
                "stage": BillStage.REGULATIONS_DRAFTED.value,
                "event_date": ev_date,
                "document_type": "UnitedKingdomDraftStatutoryInstrument",
                "citation": citation,
                "title": title,
                "note": (
                    summary[:400]
                    if summary
                    else f"Draft affirmative regulation laid under {act_title}."
                ),
                "source_api_url": f"{web_url}/data.xml",
                "web_url": web_url,
                "ingested_at": ingested_at,
            }
        )

    # 4. Parse UKSI (Commencement Orders, Substantive Regulations Laid, and Amending Instruments)
    for entry in uksi_entries:
        title = (entry.findtext("atom:title", default="", namespaces=ATOM_NS) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ATOM_NS) or "").strip()
        if not title:
            continue
        if act_norm not in _normalize_title(title) and act_norm not in _normalize_title(summary):
            continue
        ev_date = max(_extract_entry_date(entry, ra_date), ra_date)
        raw_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").replace(
            "http://", "https://"
        )
        web_url = raw_id.replace("/id/uksi/", "/uksi/")
        yr_el = entry.find("ukm:Year", ATOM_NS)
        num_el = entry.find("ukm:Number", ATOM_NS)
        alt_el = entry.find("ukm:AlternativeNumber", ATOM_NS)
        si_yr = yr_el.attrib.get("Value") if yr_el is not None else ""
        si_num = num_el.attrib.get("Value") if num_el is not None else ""
        alt_cat = alt_el.attrib.get("Category") if alt_el is not None else None
        alt_val = alt_el.attrib.get("Value") if alt_el is not None else None

        low_title = title.lower()
        is_commencement = (
            alt_cat == "C"
            or "commencement" in low_title
            or "appointed day" in low_title
        )
        is_amendment_or_revocation = any(
            kw in low_title
            for kw in ("(amendment)", "(repeal)", "(revocation)", "consequential amendments")
        )

        if is_commencement:
            c_suffix = f" (C. {alt_val})" if alt_val else ""
            citation = f"SI {si_yr}/{si_num}{c_suffix}" if si_yr and si_num else "Commencement SI"
            commencement_dates.append(ev_date)
            events.append(
                {
                    "bill_id": bill_id,
                    "stage": BillStage.COMMENCEMENT_ORDERS.value,
                    "event_date": ev_date,
                    "document_type": "CommencementOrderSI",
                    "citation": citation,
                    "title": f"{title} ({citation})",
                    "note": summary[:400]
                    if summary
                    else f"Commencement order bringing provisions of {act_title} into force.",
                    "source_api_url": f"{web_url}/made/data.xml",
                    "web_url": web_url,
                    "ingested_at": ingested_at,
                }
            )
        else:
            citation = f"SI {si_yr}/{si_num}" if si_yr and si_num else "Statutory Instrument"
            laid_reg_dates.append(ev_date)
            events.append(
                {
                    "bill_id": bill_id,
                    "stage": BillStage.REGULATIONS_LAID.value,
                    "event_date": ev_date,
                    "document_type": "UnitedKingdomStatutoryInstrument",
                    "citation": citation,
                    "title": f"{title} ({citation})",
                    "note": summary[:400]
                    if summary
                    else f"Statutory Instrument made and laid under {act_title}.",
                    "source_api_url": f"{web_url}/made/data.xml",
                    "web_url": web_url,
                    "ingested_at": ingested_at,
                }
            )

        if is_amendment_or_revocation:
            citation = f"SI {si_yr}/{si_num}" if si_yr and si_num else "Amending Instrument"
            amendment_dates.append(ev_date)
            events.append(
                {
                    "bill_id": bill_id,
                    "stage": BillStage.AMENDED_OR_REPEALED.value,
                    "event_date": ev_date,
                    "document_type": "AmendingInstrument",
                    "citation": citation,
                    "title": f"{title} ({citation})",
                    "note": summary[:400]
                    if summary
                    else f"Subsequent amendment or revocation affecting {act_title}.",
                    "source_api_url": f"{web_url}/made/data.xml",
                    "web_url": web_url,
                    "ingested_at": ingested_at,
                }
            )

    # 5. If the Act itself is marked repealed on legislation.gov.uk
    if is_repealed:
        rep_date = max(ukpga_meta.get("updated_date") or ra_date, ra_date)
        amendment_dates.append(rep_date)
        events.append(
            {
                "bill_id": bill_id,
                "stage": BillStage.AMENDED_OR_REPEALED.value,
                "event_date": rep_date,
                "document_type": "RepealedAct",
                "citation": f"{ukpga_year} c. {ukpga_number} (repealed)",
                "title": f"{act_title} — Repealed on legislation.gov.uk",
                "note": (
                    f"{act_title} ({ukpga_year} c. {ukpga_number}) is recorded as "
                    "repealed on legislation.gov.uk."
                ),
                "source_api_url": f"https://www.legislation.gov.uk/ukpga/{ukpga_year}/{ukpga_number}/data.xml",
                "web_url": f"https://www.legislation.gov.uk/ukpga/{ukpga_year}/{ukpga_number}",
                "ingested_at": ingested_at,
            }
        )

    # 6. Law in force event
    in_force_date = max(commencement_dates) if commencement_dates else ra_date
    events.append(
        {
            "bill_id": bill_id,
            "stage": BillStage.LAW_FULLY_IN_FORCE.value,
            "event_date": in_force_date,
            "document_type": "ActInForce",
            "citation": f"{ukpga_year} c. {ukpga_number}",
            "title": f"{act_title} — Commenced Provisions in Force",
            "note": (
                f"All {len(commencement_dates)} recorded Commencement Order(s) in force "
                f"as of {in_force_date.isoformat()}."
                if commencement_dates
                else (
                    f"{act_title} commenced directly via its enactment commencement "
                    f"clause on {ra_date.isoformat()}."
                )
            ),
            "source_api_url": f"https://www.legislation.gov.uk/ukpga/{ukpga_year}/{ukpga_number}/data.xml",
            "web_url": f"https://www.legislation.gov.uk/ukpga/{ukpga_year}/{ukpga_number}",
            "ingested_at": ingested_at,
        }
    )

    summary_row = {
        "bill_id": bill_id,
        "ukpga_year": ukpga_year,
        "ukpga_number": ukpga_number,
        "ukpga_uri": ukpga_uri,
        "act_title": act_title,
        "isbn": ukpga_meta["isbn"],
        "royal_assent_date": ra_date,
        "some_sections_start_date": ra_date,
        "first_commencement_order_date": min(commencement_dates) if commencement_dates else None,
        "latest_commencement_order_date": max(commencement_dates) if commencement_dates else None,
        "commencement_orders_count": len(commencement_dates),
        "first_draft_regulation_date": min(draft_reg_dates) if draft_reg_dates else None,
        "latest_draft_regulation_date": max(draft_reg_dates) if draft_reg_dates else None,
        "draft_regulations_count": len(draft_reg_dates),
        "first_laid_regulation_date": min(laid_reg_dates) if laid_reg_dates else None,
        "latest_laid_regulation_date": max(laid_reg_dates) if laid_reg_dates else None,
        "laid_regulations_count": len(laid_reg_dates),
        "law_fully_in_force_date": in_force_date,
        "is_fully_in_force": True,
        "first_amendment_or_repeal_date": min(amendment_dates) if amendment_dates else None,
        "latest_amendment_or_repeal_date": max(amendment_dates) if amendment_dates else None,
        "amendment_and_repeal_count": len(amendment_dates),
        "is_repealed": is_repealed,
        "ingested_at": ingested_at,
    }
    return summary_row, events


async def fetch_and_persist_post_royal_assent_data() -> dict[str, int]:
    """Fetch Phase 3 (After Royal Assent) data for all Government Acts and save to Neon."""
    bills_df = fetch_bills_df()
    acts_df = bills_df[bills_df["is_act"] == True].copy()  # noqa: E712

    sem = asyncio.Semaphore(25)
    headers = {"User-Agent": "westminster-weather/0.1 (legislation-ingestion)"}
    async with httpx.AsyncClient(
        timeout=25.0, follow_redirects=True, headers=headers
    ) as client:
        ukpga_by_norm = await _fetch_all_ukpga_entries(client)

        tasks = []
        for row in acts_df.itertuples(index=False):
            ra_date = row.outcome_date or row.introduced_date or date(2026, 1, 1)
            ukpga_meta = _match_bill_to_ukpga(
                int(row.bill_id), str(row.short_title), ra_date, ukpga_by_norm
            )
            tasks.append(
                _fetch_act_instruments(client, sem, int(row.bill_id), ra_date, ukpga_meta)
            )

        results = await asyncio.gather(*tasks)

    summary_rows: list[dict[str, Any]] = []
    all_events: list[dict[str, Any]] = []
    for s_row, ev_list in results:
        summary_rows.append(s_row)
        all_events.extend(ev_list)

    commencements_df = pd.DataFrame(summary_rows)
    events_df = pd.DataFrame(all_events)
    return save_post_royal_assent_tables_to_db(commencements_df, events_df)
