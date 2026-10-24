"""Stage 1 replay: for past promises, what the hazard model predicted at the time vs what
happened. Each promise's model is trained only on signals visible before its trigger date.
Read-only. Run: uv run python -m westminster_weather.takeup.replay"""

import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from westminster_weather.takeup import evaluate as ev  # noqa: E402
from westminster_weather.takeup.bill_text import first_readings  # noqa: E402
from westminster_weather.takeup.signals import TakeupData  # noqa: E402

IDEAS = [
    "nlp-00362",
    "nlp-00392",
    "nlp-00403",
    "nlp-00419",
    "nlp-00500",
    "nlp-00556",
    "nlp-00666",
    "nlp-00318",
    "nlp-00688",
]
CHECKPOINTS_WEEKS = [0, 52, 104, 208]

td = TakeupData()
T = td.tables
ideas = T["ideas"].set_index("idea_id")
links = T["idea_bill_links"].set_index("idea_id")
fr = first_readings(T["stage_sittings"])
titles = T["bills"].set_index("bill_id")["short_title"]
end = ev.default_end(td)


def row_for(df: pd.DataFrame, idea_id: str) -> pd.Series | None:
    m = df[df["signal_id"].astype(str).str.startswith(idea_id)]
    return None if m.empty else m.sort_values("signal_date").iloc[0]


out = []
for idea_id in IDEAS:
    trig = pd.Timestamp(ideas.loc[idea_id, "trigger_date"])
    bill = links.loc[idea_id, "bill_id"]
    actual_date = pd.Timestamp(fr[int(bill)]) if pd.notna(bill) else None
    actual = (
        (
            f"bill introduced {actual_date.date()} ({titles.get(int(bill), '?')}), "
            f"{(actual_date - trig).days // 7} weeks after the promise"
        )
        if actual_date is not None
        else "no Government Bill (as of data end)"
    )
    train = td.get_takeup_signals(trig)
    train = train[pd.to_datetime(train["signal_date"]) < trig]
    model = ev.fit_hazard(td, train, trig)
    print(f"\n=== {idea_id}: {ideas.loc[idea_id, 'topic'][:70]!r}")
    print(
        f"    promised {trig.date()} | trained on {len(train)} earlier promises "
        f"({train['bill_date'].notna().sum()} with bills) | REAL LIFE: {actual}"
    )
    for w in CHECKPOINTS_WEEKS:
        as_of = trig + pd.Timedelta(weeks=w)
        if as_of > end or (actual_date is not None and as_of >= actual_date):
            break
        sigs = td.get_takeup_signals(as_of)
        s = row_for(sigs, idea_id)
        if s is None:
            continue
        tl = td.mention_timeline(as_of)
        m = tl[tl["signal_id"] == s["signal_id"]]
        p = model.predict(s, m, as_of, td.sitting_dates(as_of), td.week_probability(as_of))

        def wk(x, start=trig):
            return (start + pd.Timedelta(weeks=int(x))).strftime("%b %Y") if pd.notna(x) else "—"

        hit = ""
        if actual_date is not None and pd.notna(p.get("t_10")) and pd.notna(p.get("t_90")):
            aw = (actual_date - trig).days // 7 + 1
            hit = "  ✓ inside window" if p["t_10"] <= aw <= p["t_90"] else "  ✗ outside window"
        in52 = actual_date is not None and actual_date <= as_of + pd.Timedelta(weeks=52)
        print(
            f"    as of {as_of.date()} (+{w:>3}w, {len(m)} mentions so far): "
            f"chance now {p['pi_now']:.0%}, within 52w {p['p52']:.0%} "
            f"[real: {'yes' if in52 else 'no'}], "
            f"verdict {p['verdict']}, if it comes ~{wk(p.get('t_50'))} "
            f"(80%: {wk(p.get('t_10'))}–{wk(p.get('t_90'))}){hit}"
        )
        out.append(
            {
                "idea_id": idea_id,
                "as_of": as_of.date(),
                "pi_now": p["pi_now"],
                "p52": p["p52"],
                "verdict": p["verdict"],
                "real_bill_within_52w": in52,
                "actual": actual,
            }
        )

pd.DataFrame(out).to_csv("reports/takeup/stage1_replay.csv", index=False)
print("\nwrote reports/takeup/stage1_replay.csv")
