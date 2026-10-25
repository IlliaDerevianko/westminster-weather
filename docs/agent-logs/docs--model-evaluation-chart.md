# Agent log: `docs/model-evaluation-chart`

## 2026-10-25 08:22 — docs(analysis): stakeholder chart of how the passage model's forecasts held up

- **What:** New `analysis/model_evaluation_chart.py` draws `reports/model_evaluation_for_stakeholders.png` from the existing backtest results (no re-running): left, average Royal Assent date error by the stage the forecast was made at, model vs rule of thumb (±61 vs ±83 days, closer at all 9 stages); right, deadline chances said vs happened (10 bins, test era, as served), with "said about 70%: happened 70%" and "46% more accurate than always guessing the historical average" (Brier 0.097 vs 0.180).
- **Why:** User asked for a stakeholder graph evaluating the model that highlights its strengths, without unnecessary detail.
- **Checks:** `uv run ruff check` on the module: clean. Palette validator: blue/grey separation passes for normal and colour-blind vision; the grey intentionally fails the chroma floor (neutral reference line, direct-labelled). Rendered and checked visually: no overlaps.
- **Notes:** Strengths only by design. Left out: the 80% date windows held 66% of the time (too narrow) and the overall pass chance is too pessimistic (87% predicted vs 94% passed; Brier 0.0517 vs 0.0544 for the base rate). All 10 calibration bins are shown, including 40–60% which run 11–14 points high. Commons report stage has only 15 forecasts.
