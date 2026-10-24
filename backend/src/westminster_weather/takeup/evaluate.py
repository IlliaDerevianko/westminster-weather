"""Time-split evaluation of the take-up model (TAKEUP_PLAN §2.6).

``backtest(td, cut, end)`` fits the model only on ``td.get_takeup_signals(as_of=cut)`` (ideas
triggered before ``cut``, censored at ``cut``) and tests it on ideas triggered in
``[cut, end)``, whose outcomes come from ``td.get_takeup_signals(as_of=end)``. Each test idea
is predicted at its trigger with information up to its trigger only (z features, x events on
or before the trigger, the sitting-week probabilities known at ``cut``); its bill, if any, is
hidden. Later forecasts (``forecasts_at``) are made from ``get_takeup_signals(as_of)`` itself.

Baselines: B0 = share of training ideas with a bill within H weeks; B1 = the same share by
first source (``z_src_hansard``; written statements are the reference level).

Metrics: AUC and Brier for a bill within 26 / 52 weeks of the trigger (ideas whose outcome is
known at that horizon), a 5-bin calibration table, 80% timing-window coverage, concordance
indices for timing, accuracy of 'likely' and 'unlikely' verdicts and the share 'uncertain',
sample sizes and the parameter table.

``run_evaluation(write)`` runs both splits plus a fit on all ideas, writes the four charts and
``metrics.json`` to ``backend/reports/takeup/`` and ``docs/model/TAKEUP_EVALUATION.md``, and with
``write=True`` saves the run and its current forecasts to ``takeup_runs`` / ``takeup_forecasts``
on the take-up database (never production; ``takeup.db`` refuses it).
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from westminster_weather.takeup import model as tm

log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[3]
REPORT_DIR = BACKEND_DIR / "reports" / "takeup"
DOC_PATH = BACKEND_DIR.parent / "docs" / "model" / "TAKEUP_EVALUATION.md"

SPLITS = (date(2019, 1, 1), date(2022, 1, 1))
HORIZONS = (26, 52)
VERDICT_WEEKS = 104  # an idea silent this long counts as 'not taken up' for verdict accuracy
CALIBRATION_BINS = 5
SOURCE_COL = "z_src_hansard"
FORECAST_KEYS = (
    "pi",
    "pi_lo",
    "pi_hi",
    "verdict",
    "weeks_elapsed",
    "pi_now",
    "p26",
    "p52",
    "t_10",
    "t_50",
    "t_90",
)

# light-theme categorical palette (dataviz reference instance)
BLUE, ORANGE, AQUA, INK, MUTED, GRID = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#0b0b0b",
    "#52514e",
    "#e4e3df",
)


class SignalSource(Protocol):
    """What the evaluation needs from ``signals.TakeupData`` (tests pass a fake)."""

    def get_takeup_signals(self, as_of: Any, features: list[str] | None = None) -> pd.DataFrame:
        """One row per signal visible at ``as_of`` (DATA_CONTRACT §10.9)."""

    def sitting_dates(self, as_of: Any) -> np.ndarray:
        """Dates either House sat, up to ``as_of``."""

    def week_probability(self, as_of: Any) -> pd.Series:
        """P(sitting) by ISO week, from sittings up to ``as_of``."""


# ----------------------------------------------------------------------- metrics


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("z_", "x_"))]


def outcome_labels(d: dict[str, Any], H: int) -> tuple[np.ndarray, np.ndarray]:
    """(known, y) for 'bill within H weeks of the trigger' from prepared data.

    Known if a bill was seen (at any week) or the idea was followed for at least H weeks;
    y = a bill in weeks 1..H.
    """
    bill = d["delta"] == 1
    known = bill | (d["weeks_observed"] >= H)
    return known, bill & (d["t"] <= H)


def binary_metrics(y: Sequence[int], p: Sequence[float]) -> dict[str, Any]:
    """Sample size, positives, AUC (None with one class) and Brier score."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    n = len(y)
    auc = float(roc_auc_score(y, p)) if n and 0 < y.sum() < n else None
    brier = float(np.mean((p - y) ** 2)) if n else None
    return {"n": n, "n_bill": int(y.sum()), "auc": auc, "brier": brier}


def calibration_table(
    y: Sequence[int], p: Sequence[float], bins: int = CALIBRATION_BINS
) -> list[dict[str, Any]]:
    """Equal-width bins of predicted probability: n, mean predicted, observed share."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        out.append(
            {
                "bin": f"{edges[b]:.0%}-{edges[b + 1]:.0%}",
                "lo": float(edges[b]),
                "hi": float(edges[b + 1]),
                "n": int(m.sum()),
                "mean_predicted": float(p[m].mean()) if m.any() else None,
                "observed": float(y[m].mean()) if m.any() else None,
            }
        )
    return out


def window_coverage(t_bill: Sequence[int], t_lo: Sequence[Any], t_hi: Sequence[Any]) -> dict:
    """Share of bills whose week lies in [t_10, t_90]; a missing t_90 means beyond 5 years."""
    hits = []
    for t, lo, hi in zip(t_bill, t_lo, t_hi, strict=True):
        lo = tm.T_MAX + 1 if lo is None or pd.isna(lo) else lo
        hi = np.inf if hi is None or pd.isna(hi) else hi
        hits.append(lo <= t <= hi)
    return {"n": len(hits), "coverage": float(np.mean(hits)) if hits else None}


def concordance(times: Sequence[float], scores: Sequence[float], events: Sequence[int]) -> dict:
    """Harrell's C (lifelines); higher ``scores`` = later (or no) bill. Censored rows only
    count against rows with an earlier observed bill."""
    from lifelines.utils import concordance_index

    times, events = np.asarray(times, dtype=float), np.asarray(events, dtype=int)
    n = len(times)
    if n < 2 or events.sum() == 0:
        return {"n": n, "c_index": None}
    try:
        c = float(concordance_index(times, np.asarray(scores, dtype=float), events))
    except ZeroDivisionError:  # no comparable pairs
        c = None
    return {"n": n, "n_bill": int(events.sum()), "c_index": c}


def verdict_accuracy(verdicts: Sequence[str], ever: Sequence[int]) -> dict[str, Any]:
    """Accuracy of 'likely' (bill seen) and 'unlikely' (no bill) separately, share uncertain."""
    v, y = np.asarray(verdicts, dtype=object), np.asarray(ever, dtype=int)
    out: dict[str, Any] = {"n": len(v)}
    for name, want in (("likely", 1), ("unlikely", 0)):
        m = v == name
        out[f"n_{name}"] = int(m.sum())
        out[f"{name}_accuracy"] = float(np.mean(y[m] == want)) if m.any() else None
    out["n_uncertain"] = int((v == "uncertain").sum())
    out["share_uncertain"] = float(out["n_uncertain"] / len(v)) if len(v) else None
    return out


def baseline_rates(train: pd.DataFrame, d: dict[str, Any], H: int) -> dict[str, Any]:
    """B0 (overall) and B1 (by first source) shares of a bill within H weeks, among training
    ideas whose H-week outcome is known at the cut."""
    known, y = outcome_labels(d, H)
    b0 = float(y[known].mean()) if known.any() else 0.0
    b1 = {}
    src = train[SOURCE_COL].to_numpy(float) if SOURCE_COL in train else np.zeros(len(train))
    for level, name in ((0.0, "written_statement"), (1.0, "hansard")):
        m = known & (src == level)
        b1[name] = {"n": int(m.sum()), "rate": float(y[m].mean()) if m.any() else b0}
    return {"H": H, "n": int(known.sum()), "b0": b0, "b1": b1}


def apply_baselines(test: pd.DataFrame, rates: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """B0 and B1 predictions for the rows of ``test``."""
    n = len(test)
    b0 = np.full(n, rates["b0"])
    src = test[SOURCE_COL].to_numpy(float) if SOURCE_COL in test else np.zeros(n)
    b1 = np.where(
        src == 1.0, rates["b1"]["hansard"]["rate"], rates["b1"]["written_statement"]["rate"]
    )
    return b0, b1


# -------------------------------------------------------------------- predictions


def _predict_rows(model: tm.TakeUpModel, d: dict[str, Any], horizons=HORIZONS) -> list[dict]:
    out = []
    for k in range(len(d["t"])):
        p = model.predict(**tm.row_inputs(d, k), horizons=horizons)
        out.append({key: p[key] for key in p if not key.endswith("_sitting")})
    return out


def predict_at_trigger(
    model: tm.TakeUpModel,
    df: pd.DataFrame,
    sitting_dates: Any,
    week_probability: pd.Series | None,
) -> pd.DataFrame:
    """Forecast every row of ``df`` as of its own ``signal_date``.

    The row's bill is hidden and its ``x_*`` events after the trigger are ignored (prepare
    drops events after the row's ``data_end``); weeks after the trigger use the sitting-week
    probabilities. So nothing after the trigger can change the forecast.
    """
    if df.empty:
        return pd.DataFrame()
    view = df.copy()
    view["bill_date"] = pd.NaT
    view["data_end"] = view["signal_date"]
    d = tm.prepare(
        view,
        sitting_dates,
        week_probability,
        data_end=view["signal_date"].max(),
    )
    preds = pd.DataFrame(_predict_rows(model, d))
    preds.insert(0, "signal_id", df["signal_id"].astype(str).to_numpy())
    preds.insert(1, "as_of", pd.to_datetime(df["signal_date"]).dt.date.to_numpy())
    return preds


def forecasts_at(
    model: tm.TakeUpModel,
    td: SignalSource,
    as_of: Any,
    features: list[str],
    week_probability: pd.Series | None = None,
    since: Any = None,
) -> pd.DataFrame:
    """Forecasts for every signal still silent at ``as_of`` (no bill yet), from
    ``get_takeup_signals(as_of)`` only. ``since``: keep signals triggered on or after it.
    Columns: signal_id, idea_id, signal_date, data_end, as_of and the predict() outputs."""
    as_of = pd.Timestamp(as_of).normalize()
    df = td.get_takeup_signals(as_of, features=features)
    df = df[df["bill_date"].isna()]
    if since is not None:
        df = df[df["signal_date"] >= pd.Timestamp(since)]
    df = df.reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()
    wp = td.week_probability(as_of) if week_probability is None else week_probability
    d = tm.prepare(df, td.sitting_dates(as_of), wp, data_end=as_of, calendar_end=as_of)
    preds = pd.DataFrame(_predict_rows(model, d))
    meta = pd.DataFrame(
        {
            "signal_id": df["signal_id"].astype(str),
            "idea_id": df["idea_id"].astype(str) if "idea_id" in df else df["signal_id"],
            "signal_date": df["signal_date"],
            "data_end": df["data_end"] if "data_end" in df else as_of,
            "as_of": as_of.date(),
        }
    )
    return pd.concat([meta, preds], axis=1)


# ----------------------------------------------------------------------- backtest


def _assert_no_leakage(train: pd.DataFrame, cut: pd.Timestamp) -> None:
    """Training rows must be triggered before the cut and show nothing dated after it."""
    assert (train["signal_date"] < cut).all(), "training signal after the cut"
    for col in ["bill_date", "data_end", *[c for c in train if c.startswith("x_")]]:
        if col in train:
            dates = pd.to_datetime(train[col])
            assert not (dates > cut).any(), f"training {col} after the cut"


def backtest(
    td: SignalSource,
    cut: Any,
    end: Any,
    horizons: Sequence[int] = HORIZONS,
    later_points: bool = True,
) -> dict[str, Any]:
    """Fit on ideas triggered before ``cut`` (seen as of ``cut``), test on ideas triggered in
    ``[cut, end)`` (outcomes as of ``end``). Returns the model, the per-idea test table and
    the metrics."""
    cut, end = pd.Timestamp(cut).normalize(), pd.Timestamp(end).normalize()
    if not cut < end:
        raise ValueError("cut must be before end")

    train = td.get_takeup_signals(cut)
    train = train[train["signal_date"] < cut].reset_index(drop=True)
    _assert_no_leakage(train, cut)
    if train.empty:
        raise ValueError(f"no training ideas before {cut.date()}")
    features = feature_columns(train)
    wp = td.week_probability(cut)
    d_train = tm.prepare(train, td.sitting_dates(cut), wp, data_end=cut, calendar_end=cut)
    model = tm.TakeUpModel().fit(d_train)

    full = td.get_takeup_signals(end, features=features)
    test = full[(full["signal_date"] >= cut) & (full["signal_date"] < end)].reset_index(drop=True)
    if test.empty:
        raise ValueError(f"no test ideas in [{cut.date()}, {end.date()})")
    d_test = tm.prepare(test, td.sitting_dates(end), td.week_probability(end), data_end=end)
    # Prediction uses sitting dates only up to each trigger (observed weeks = 0), so passing
    # the calendar to `end` adds nothing beyond `wp`, which is known at the cut.
    preds = predict_at_trigger(model, test, td.sitting_dates(end), wp)

    table = pd.DataFrame(
        {
            "signal_id": test["signal_id"].astype(str),
            "idea_id": test["idea_id"].astype(str) if "idea_id" in test else test["signal_id"],
            "title": test["title"] if "title" in test else "",
            "signal_date": test["signal_date"],
            "bill": d_test["delta"] == 1,
            "t": d_test["t"],
            "weeks_observed": d_test["weeks_observed"],
            SOURCE_COL: test[SOURCE_COL] if SOURCE_COL in test else 0.0,
        }
    )
    for key in ("pi", "pi_lo", "pi_hi", "verdict", "p26", "p52", "t_10", "t_50", "t_90"):
        if key in preds:
            table[key] = preds[key].to_numpy()

    metrics: dict[str, Any] = {
        "cut": str(cut.date()),
        "end": str(end.date()),
        "n_train": len(train),
        "n_train_bills": int(d_train["delta"].sum()),
        "n_test": len(test),
        "n_test_bills": int(d_test["delta"].sum()),
        "features": features,
        "converged": model.converged,
        "horizons": {},
    }
    for H in horizons:
        known, y = outcome_labels(d_test, H)
        rates = baseline_rates(train, d_train, H)
        b0, b1 = apply_baselines(test, rates)
        p = table[f"p{H}"].to_numpy(float)
        table[f"b0_{H}"], table[f"b1_{H}"] = b0, b1
        table[f"known_{H}"], table[f"y_{H}"] = known, y
        metrics["horizons"][str(H)] = {
            "model": binary_metrics(y[known], p[known]),
            "b0": binary_metrics(y[known], b0[known]),
            "b1": binary_metrics(y[known], b1[known]),
            "baseline_rates": rates,
            "calibration": calibration_table(y[known], p[known]),
        }

    bills = table["bill"].to_numpy()
    metrics["timing_window"] = window_coverage(
        table.loc[bills, "t"], table.loc[bills, "t_10"], table.loc[bills, "t_90"]
    )
    metrics["concordance"] = {
        # all ideas: higher 1 - p52 = later or no bill; censored at weeks observed
        "all_ideas_p52": concordance(table["t"], 1 - table["p52"], bills.astype(int)),
        # taken-up ideas only: predicted median week vs actual week
        "bills_t50": concordance(
            table.loc[bills, "t"],
            table.loc[bills, "t_50"].astype(float).fillna(tm.T_MAX + 1),
            np.ones(int(bills.sum()), dtype=int),
        ),
    }
    known_ever = bills | (table["weeks_observed"].to_numpy() >= VERDICT_WEEKS)
    metrics["verdicts"] = {
        "known_after_weeks": VERDICT_WEEKS,
        **verdict_accuracy(table.loc[known_ever, "verdict"], bills[known_ever].astype(int)),
    }
    metrics["parameters"] = _parameter_records(model)
    if later_points:
        metrics["later_as_of"] = _later_points(model, td, features, wp, cut, end, full)
    return {
        "cut": cut,
        "end": end,
        "model": model,
        "train": train,
        "test": table,
        "metrics": metrics,
    }


def _later_points(
    model: tm.TakeUpModel,
    td: SignalSource,
    features: list[str],
    wp: pd.Series,
    cut: pd.Timestamp,
    end: pd.Timestamp,
    full: pd.DataFrame,
    H: int = 52,
) -> dict[str, Any]:
    """p52 made at each 1 January after the cut (for test ideas still silent then), scored on
    whether a bill came within 52 weeks of that date."""
    by_sid = full.drop_duplicates("signal_id").set_index("signal_id")
    by_idea = full.drop_duplicates("idea_id").set_index("idea_id") if "idea_id" in full else None
    ys, ps, points = [], [], []
    year = cut.year + 1
    while pd.Timestamp(year, 1, 1) + pd.Timedelta(weeks=H) <= end:
        as_of = pd.Timestamp(year, 1, 1)
        year += 1
        f = forecasts_at(model, td, as_of, features, wp, since=cut)
        if f.empty:
            continue
        f = f[pd.to_datetime(f["data_end"]) >= as_of]  # not censored by a dissolution
        n = 0
        for _, r in f.iterrows():
            if r["signal_id"] in by_sid.index:
                row = by_sid.loc[r["signal_id"]]
            elif by_idea is not None and r["idea_id"] in by_idea.index:
                row = by_idea.loc[r["idea_id"]]
            else:
                continue
            bill, stop = row["bill_date"], pd.Timestamp(row["data_end"])
            hit = pd.notna(bill) and as_of < bill <= as_of + pd.Timedelta(weeks=H)
            if hit or stop >= as_of + pd.Timedelta(weeks=H):
                ys.append(int(hit))
                ps.append(float(r[f"p{H}"]))
                n += 1
        points.append({"as_of": str(as_of.date()), "n": n})
    return {"H": H, "points": points, **binary_metrics(ys, ps)}


def _parameter_records(model: tm.TakeUpModel) -> list[dict[str, Any]]:
    tab = model.parameter_table()
    return [
        {
            "parameter": r.parameter,
            "estimate": float(r.estimate),
            "std_err": None if pd.isna(r.std_err) else float(r.std_err),
        }
        for r in tab.itertuples()
    ]


# --------------------------------------------------------------------- full fit


def fit_all(td: SignalSource, end: Any) -> dict[str, Any]:
    """Fit on every idea visible at ``end`` (for feature effects and current forecasts)."""
    end = pd.Timestamp(end).normalize()
    df = td.get_takeup_signals(end).reset_index(drop=True)
    wp = td.week_probability(end)
    d = tm.prepare(df, td.sitting_dates(end), wp, data_end=end, calendar_end=end)
    model = tm.TakeUpModel().fit(d)
    features = feature_columns(df)
    forecasts = forecasts_at(model, td, end, features, wp)
    return {"end": end, "model": model, "signals": df, "prepared": d, "forecasts": forecasts}


# ------------------------------------------------------------------------ charts


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=130)
    fig.patch.set_facecolor("white")
    return plt, fig, ax


def plot_calibration(backtests: list[dict[str, Any]], path: Path, H: int = 52) -> Path:
    plt, fig, ax = _figure()
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect")
    for bt, color in zip(backtests, (BLUE, ORANGE, AQUA), strict=False):
        rows = [r for r in bt["metrics"]["horizons"][str(H)]["calibration"] if r["n"]]
        x = [r["mean_predicted"] for r in rows]
        y = [r["observed"] for r in rows]
        size = [20 + 4 * np.sqrt(r["n"]) * 4 for r in rows]
        label = f"trained before {bt['metrics']['cut'][:4]}"
        ax.plot(x, y, color=color, linewidth=2, label=label)
        ax.scatter(x, y, s=size, color=color, edgecolor="white", linewidth=2, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(
        ax,
        f"Calibration: bill within {H} weeks of the trigger (test ideas)",
        "predicted chance",
        "share that got a bill",
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def pick_example_ideas(d: dict[str, Any], min_weeks: int = 8) -> dict[str, int]:
    """Row indices of a fast bill, a slow bill and the longest-silent idea."""
    t, delta, obs = d["t"], d["delta"], d["weeks_observed"]
    out = {}
    bills = np.nonzero((delta == 1) & (t >= min_weeks))[0]
    if len(bills):
        out["fast"] = int(bills[np.argmin(t[bills])])
        out["slow"] = int(bills[np.argmax(t[bills])])
    silent = np.nonzero(delta == 0)[0]
    if len(silent):
        out["never"] = int(silent[np.argmax(obs[silent])])
    return out


def pi_now_curve(model: tm.TakeUpModel, d: dict[str, Any], k: int) -> tuple[np.ndarray, np.ndarray]:
    """pi_now after s = 0..S silent weeks (S = bill week - 1, or weeks observed)."""
    inputs = tm.row_inputs(d, k)
    last = int(d["t"][k]) - 1 if d["delta"][k] == 1 else int(d["weeks_observed"][k])
    weeks = np.arange(0, max(last, 0) + 1)
    vals = [
        model.predict(inputs["z"], inputs["xweek_row"], int(s), (52,), inputs["mask"])["pi_now"]
        for s in weeks
    ]
    return weeks, np.asarray(vals)


def plot_pi_now(model: tm.TakeUpModel, d: dict[str, Any], path: Path) -> Path:
    plt, fig, ax = _figure()
    titles = d["titles"] if d.get("titles") is not None else d["ids"]
    colors = {"fast": AQUA, "slow": BLUE, "never": ORANGE}
    for kind, k in pick_example_ideas(d).items():
        weeks, vals = pi_now_curve(model, d, k)
        name = str(titles[k])[:40]
        label = f"{kind}: {name}"
        if d["delta"][k] == 1:
            label += f" (bill week {d['t'][k]})"
        ax.plot(weeks, vals, color=colors[kind], linewidth=2, label=label)
        ax.scatter(weeks[-1:], vals[-1:], s=40, color=colors[kind], edgecolor="white", zorder=3)
    ax.set_ylim(0, 1)
    _style(
        ax,
        "Chance an idea is ever taken up, updated each silent week",
        "weeks since the trigger",
        "pi_now",
    )
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_time_to_bill(d: dict[str, Any], path: Path) -> Path:
    plt, fig, ax = _figure()
    t = d["t"][d["delta"] == 1]
    bins = np.arange(0, max(int(t.max()) if len(t) else 52, 52) + 14, 13)
    ax.hist(t, bins=bins, color=BLUE, edgecolor="white", linewidth=2)
    for H in HORIZONS:
        ax.axvline(H, color=MUTED, linewidth=1, linestyle="--")
        ax.text(H, ax.get_ylim()[1] * 0.95, f" {H} wk", color=MUTED, fontsize=8, va="top")
    _style(
        ax,
        f"Weeks from trigger to first reading ({len(t)} ideas taken up)",
        "calendar weeks (13-week bins)",
        "ideas",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_feature_effects(model: tm.TakeUpModel, path: Path) -> Path:
    plt, fig, ax = _figure()
    tab = model.parameter_table()
    tab = tab[tab["parameter"].str.startswith(("z_", "x_"))].iloc[::-1].reset_index(drop=True)
    y = np.arange(len(tab))
    se = tab["std_err"].fillna(0).to_numpy()
    colors = [BLUE if p.startswith("z_") else ORANGE for p in tab["parameter"]]
    for i, r in tab.iterrows():
        ax.errorbar(
            r["estimate"],
            i,
            xerr=1.96 * se[i],
            fmt="o",
            color=colors[i],
            markersize=6,
            elinewidth=2,
            capsize=0,
        )
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(tab["parameter"], fontsize=8)
    _style(
        ax,
        "Feature effects (log-odds, 95% CI)",
        "blue z: chance of take-up   |   orange x: weekly hazard once switched on",
        "",
    )
    fig.set_size_inches(7, max(3.0, 0.35 * len(tab) + 1.5))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def make_charts(
    backtests: list[dict[str, Any]], full: dict[str, Any], out_dir: Path
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    d, model = full["prepared"], full["model"]
    paths = {
        "calibration": plot_calibration(backtests, out_dir / "calibration.png"),
        "pi_now_curves": plot_pi_now(model, d, out_dir / "pi_now_curves.png"),
        "time_to_bill": plot_time_to_bill(d, out_dir / "time_to_bill.png"),
        "feature_effects": plot_feature_effects(model, out_dir / "feature_effects.png"),
    }
    return {k: str(v) for k, v in paths.items()}


# ------------------------------------------------------------------------ report


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, pd.Timestamp | date):
        return str(x.date() if isinstance(x, pd.Timestamp) else x)
    return x


def _pct(x: Any) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _num(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def write_report(metrics: dict[str, Any], path: Path = DOC_PATH) -> Path:
    """Plain-English evaluation report (docs/model/TAKEUP_EVALUATION.md) from ``metrics``."""
    full = metrics["full_fit"]
    feats = ", ".join(f"`{f}`" for f in full["features"]) or "none"
    lines = [
        "# Take-up model: evaluation",
        "",
        f"Generated {metrics['generated_at'][:10]} by `takeup/evaluate.py` "
        f"(code {metrics['code_version'][:10]}). Charts and `metrics.json` are in "
        "`backend/reports/takeup/`.",
        "",
        "The question: when a minister commits to legislate on something, will a Government Bill "
        "on it get a first reading, and when? See `TAKEUP_PLAN.md` for the model.",
        "",
        "## Data",
        "",
        f"- Ideas visible at {full['end']}: **{full['n_ideas']}**, of which **{full['n_bills']}** "
        f"already have a bill ({_pct(full['n_bills'] / max(full['n_ideas'], 1))}).",
        f"- Features used in the full fit: {feats}.",
    ]
    if full.get("dropped_features"):
        dropped = ", ".join(f"`{k}` ({v})" for k, v in full["dropped_features"].items())
        lines.append(f"- Dropped for having fewer than 15 positive cases: {dropped}.")
    lines += ["", "## Time-split backtests", ""]
    lines += [
        "Each split trains only on what was visible at the cut (ideas triggered before it, "
        "followed up to it) and tests on ideas triggered after it, each predicted at its own "
        "trigger date. A test idea counts for a horizon only if its outcome is known (a bill "
        "came, or it was followed for the whole horizon).",
        "",
    ]
    for bt in metrics["backtests"]:
        lines += [
            f"### Trained before {bt['cut']}, tested {bt['cut']} to {bt['end']}",
            "",
            f"Training: {bt['n_train']} ideas ({bt['n_train_bills']} with a bill by the cut). "
            f"Test: {bt['n_test']} ideas ({bt['n_test_bills']} with a bill by {bt['end']}).",
            "",
            "| Horizon | n | bills | Model AUC | Model Brier | B0 Brier | B1 AUC | B1 Brier |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for H, h in bt["horizons"].items():
            m, b0, b1 = h["model"], h["b0"], h["b1"]
            lines.append(
                f"| {H} weeks | {m['n']} | {m['n_bill']} | {_num(m['auc'])} | "
                f"{_num(m['brier'])} | {_num(b0['brier'])} | {_num(b1['auc'])} | "
                f"{_num(b1['brier'])} |"
            )
        r52 = bt["horizons"].get("52", {}).get("baseline_rates")
        if r52:
            lines += [
                "",
                f"B0 predicts {_pct(r52['b0'])} for everyone; B1 predicts "
                f"{_pct(r52['b1']['written_statement']['rate'])} for ideas first seen in a "
                f"written statement and {_pct(r52['b1']['hansard']['rate'])} for Hansard "
                "(52 weeks). B0 has no ranking, so its AUC is 0.5 by definition.",
            ]
        cal = bt["horizons"].get("52", {}).get("calibration", [])
        if cal:
            lines += ["", "Calibration, bill within 52 weeks:", ""]
            lines += ["| Predicted | n | mean predicted | actual |", "|---|---|---|---|"]
            for r in cal:
                lines.append(
                    f"| {r['bin']} | {r['n']} | {_pct(r['mean_predicted'])} | "
                    f"{_pct(r['observed'])} |"
                )
        tw, cc, v = bt["timing_window"], bt["concordance"], bt["verdicts"]
        lines += [
            "",
            f"- 80% timing window (t_10 to t_90) contained the actual week for "
            f"{_pct(tw['coverage'])} of {tw['n']} bills (target 80%).",
            f"- Concordance: all ideas ranked by p52 {_num(cc['all_ideas_p52']['c_index'])}; "
            f"bills ranked by predicted median week {_num(cc['bills_t50']['c_index'])} "
            "(0.5 = chance).",
            f"- Verdicts (ideas with a bill or silent for {v['known_after_weeks']}+ weeks, "
            f"n = {v['n']}): 'likely' {v['n_likely']} ({_pct(v['likely_accuracy'])} got a bill), "
            f"'unlikely' {v['n_unlikely']} ({_pct(v['unlikely_accuracy'])} did not), "
            f"'uncertain' {_pct(v['share_uncertain'])}.",
        ]
        later = bt.get("later_as_of")
        if later and later["n"]:
            lines.append(
                f"- Forecasts on later 1 Januaries for still-silent ideas (n = {later['n']}): "
                f"AUC {_num(later['auc'])}, Brier {_num(later['brier'])} for a bill within 52 "
                "weeks."
            )
        lines.append("")

    lines += [
        "## Parameters (fit on all ideas)",
        "",
        "| Parameter | Estimate | Std. error |",
        "|---|---|---|",
    ]
    for p in full["parameters"]:
        lines.append(f"| `{p['parameter']}` | {p['estimate']:.3f} | {_num(p['std_err'])} |")
    lines += [
        "",
        "`z_*` move the chance an idea is ever taken up; `x_*` move the weekly chance of a "
        "first reading once the event has happened; `alpha_weeks_*` are the baseline weekly "
        "log-odds by weeks since the trigger.",
        "",
        "## Weak spots and limitations",
        "",
    ]
    lines += [f"- {w}" for w in metrics.get("weak_spots", [])]
    lines += [
        "- King's Speech data is not used (decision of 2026-10-24); ideas come only from "
        "ministers' words in written statements and Hansard.",
        "- Commitments, topics and links come from rules-only NLP (phrase lexicon, TF-IDF "
        "grouping, title similarity); wrong groupings or links show up here as label noise.",
        "- An idea with no linked bill is treated as not (yet) taken up, so a missed link is "
        "counted as a failure of the idea.",
        "- Merged ideas take the earliest trigger of their groups, which uses knowledge of the "
        "shared bill; test ideas that later merged may look slightly earlier than they were.",
        "- Outcomes after a change of government are censored at the dissolution, so the test "
        "sets after 2024 are short.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def weak_spots(metrics: dict[str, Any]) -> list[str]:
    """Automatic notes on where the model is no better than the baselines."""
    notes = []
    for bt in metrics["backtests"]:
        for H, h in bt["horizons"].items():
            m, b1 = h["model"], h["b1"]
            if m["n"] < 30:
                notes.append(f"Split {bt['cut']}, {H} weeks: only {m['n']} test ideas.")
            if m["brier"] is not None and b1["brier"] is not None and m["brier"] >= b1["brier"]:
                notes.append(
                    f"Split {bt['cut']}, {H} weeks: model Brier {m['brier']:.3f} is no better "
                    f"than B1 ({b1['brier']:.3f})."
                )
        cov = bt["timing_window"]["coverage"]
        if cov is not None and abs(cov - 0.8) > 0.1:
            notes.append(f"Split {bt['cut']}: 80% timing window covers {cov:.0%} of bills.")
    return notes


# -------------------------------------------------------------------------- runs


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def save_run(
    engine: Any,
    model: tm.TakeUpModel,
    train_cut: Any,
    metrics: dict[str, Any],
    forecasts: pd.DataFrame,
    version: str | None = None,
) -> int:
    """Insert one ``takeup_runs`` row and its ``takeup_forecasts`` (keyed by run_id,
    signal_id, as_of) through ``takeup.db.upsert_rows``; returns the run_id. Refuses the
    production database."""
    from sqlalchemy import func, select

    from westminster_weather.takeup import db
    from westminster_weather.takeup.tables import takeup_runs

    now = datetime.now(UTC)
    with engine.begin() as conn:
        db._refuse_production(conn)
        run_id = int(
            conn.execute(select(func.coalesce(func.max(takeup_runs.c.run_id), 0))).scalar()
        )
        run_id += 1
        params = model.parameter_table()
        db.upsert_rows(
            conn,
            "takeup_runs",
            [
                {
                    "run_id": run_id,
                    "created_at": now,
                    "code_version": version or code_version(),
                    "train_cut": pd.Timestamp(train_cut).date(),
                    "features": ",".join([*model.zcols, *model.xcols]),
                    "parameters": json.dumps(
                        _jsonable(
                            {
                                r.parameter: {"estimate": r.estimate, "std_err": r.std_err}
                                for r in params.itertuples()
                            }
                        )
                    ),
                    "metrics": json.dumps(_jsonable(metrics)),
                }
            ],
        )
        rows = []
        for r in forecasts.to_dict("records"):
            row = {
                "run_id": run_id,
                "signal_id": str(r["signal_id"]),
                "idea_id": str(r.get("idea_id") or str(r["signal_id"]).split(":")[0]),
                "as_of": pd.Timestamp(r["as_of"]).date(),
                "created_at": now,
            }
            for key in FORECAST_KEYS:
                v = r.get(key)
                if key.startswith("t_") or key == "weeks_elapsed":
                    v = None if v is None or pd.isna(v) else int(v)
                elif key != "verdict":
                    v = float(v)
                row[key] = v
            rows.append(row)
        db.upsert_rows(conn, "takeup_forecasts", rows)
    return run_id


# ---------------------------------------------------------------------- entry point


def default_end(td: SignalSource) -> pd.Timestamp:
    """Last sitting day on record (outcomes are not observed after the calendar ends)."""
    sit = td.sitting_dates(pd.Timestamp.today())
    return pd.Timestamp(sit.max()) if len(sit) else pd.Timestamp.today().normalize()


def run_evaluation(
    write: bool = False,
    td: SignalSource | None = None,
    end: Any = None,
    splits: Sequence[Any] = SPLITS,
    out_dir: Path = REPORT_DIR,
    doc_path: Path | None = DOC_PATH,
    engine: Any = None,
) -> dict[str, Any]:
    """Backtests for ``splits``, a fit on all ideas, charts, metrics.json and the report.

    ``write=True`` also saves the full-fit run and its forecasts at ``end`` to the take-up
    database (``engine``, default ``takeup.db.get_takeup_engine()``). Returns the metrics,
    plus ``run_id`` when written.
    """
    if td is None:
        from westminster_weather.takeup.signals import TakeupData

        td = TakeupData()
    end = default_end(td) if end is None else pd.Timestamp(end).normalize()
    if td.get_takeup_signals(end).empty:
        raise ValueError("no ideas in the take-up database yet; run the NLP pipeline first")

    backtests = []
    for cut in splits:
        try:
            backtests.append(backtest(td, cut, end))
        except ValueError as exc:
            log.warning("skipping split %s: %s", cut, exc)
    full = fit_all(td, end)
    d = full["prepared"]
    metrics: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "code_version": code_version(),
        "backtests": [bt["metrics"] for bt in backtests],
        "full_fit": {
            "end": str(end.date()),
            "n_ideas": len(full["signals"]),
            "n_bills": int(d["delta"].sum()),
            "features": feature_columns(full["signals"]),
            "dropped_features": dict(getattr(td, "dropped_features", {}) or {}),
            "converged": full["model"].converged,
            "parameters": _parameter_records(full["model"]),
            "n_forecasts": len(full["forecasts"]),
        },
    }
    metrics["weak_spots"] = weak_spots(metrics)
    metrics["charts"] = make_charts(backtests, full, Path(out_dir)) if backtests else {}
    metrics = _jsonable(metrics)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if doc_path is not None:
        write_report(metrics, Path(doc_path))
    if write:
        from westminster_weather.takeup import db

        eng = engine if engine is not None else db.get_takeup_engine()
        metrics["run_id"] = save_run(eng, full["model"], end, metrics, full["forecasts"])
    return metrics
