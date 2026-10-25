# Stage duration correlation

**Question:** if a UK Government Bill moves slowly through earlier Parliamentary stages, does it
also move slowly through later ones?

Regenerate everything in this folder from `backend/`:

```bash
uv run python -m westminster_weather.analysis.stage_durations
```

## Data

- **Bills:** 438 Commons-first Government Bills (sessions 2006–07 onwards). The 150 Lords-first
  bills are left out so that "earlier" and "later" mean the same stages for every bill.
- **Segments sampled:** five steps spread across the journey, plus totals for each House.
  Gaps that are usually the same day (e.g. Commons report → 3rd reading) are not used.

  | Segment | Median (calendar days) |
  |---|---|
  | Commons 1st → 2nd reading | 13 |
  | Commons 2nd → 3rd reading (committee + report) | 49 |
  | Lords 1st reading → committee | 41 |
  | Lords committee → report | 29 |
  | Lords report → Royal Assent (incl. ping-pong) | 14 |
  | Whole Commons (1st → 3rd reading) | 74 |
  | Lords 1st reading → Royal Assent | 64 |

- **Measure:** Spearman rank correlation (ρ) of calendar days. It compares rankings, so the very
  long durations of a few bills don't dominate. Sitting days, which leave out recesses, give
  the same picture (see `metrics.json`).
- **Two groups:** *all bills*, and *regular bills*. Regular bills exclude the 108 bills that
  cleared the whole Commons in 14 days or less: emergency, finance and end-of-Parliament
  "wash-up" bills, which are fast at every stage.

## Findings

1. **Across all bills, the correlation is strong (ρ 0.40–0.75).** Most of that comes from the
   fast-tracked group, which is fast everywhere.
2. **For regular bills, the start of a bill tells you almost nothing.** Commons 1st → 2nd
   reading time correlates with later stages at only ρ ≈ 0.1.
3. **Committee-heavy stages predict each other moderately.** Time in Commons committee and
   report (2nd → 3rd reading) correlates with Lords committee → report (ρ = 0.42) and with
   Lords report → Royal Assent (ρ = 0.35). The closer two stages are, the stronger the
   link: Lords committee → report vs Lords report → Royal Assent gives ρ = 0.47.
4. **Whole Commons vs whole Lords:** ρ = 0.75 for all bills, and **ρ = 0.46 for regular
   bills**. A bill that takes long in the Commons tends to take long in the Lords too.
5. **A slow start barely changes the chance of passing.** 98% of the fastest quarter (by
   Commons 1st → 2nd reading) reached Royal Assent, against 91% of the slowest quarter.

## Charts

| File | What it shows |
|---|---|
| `1_correlation_heatmap.png` | Correlation of every earlier segment with every later one, for all bills (left) and regular bills (right). Redder = stronger link. |
| `2_stage_pairs_scatter.png` | Each step against the next, plus whole Commons vs whole Lords. Grey points are fast-tracked bills; the orange line is the median of the later step for each quarter of the earlier step. |
| `3_passage_by_early_speed.png` | Share of bills reaching Royal Assent, by how fast they went from Commons 1st to 2nd reading. |

`stage_durations_per_bill.csv` has every bill's durations (calendar and sitting days) and its
fast-tracked flag. `metrics.json` has all coefficients, p-values and counts.

## Caveats

- **Survivorship:** a pair of segments only includes bills that reached both, so bills that
  fell early only count in their early segments.
- **Not controlled for bill size or session timing:** `size_pages` and `size_clauses` are
  empty in the database. Long bills introduced early in a session may simply have more time
  at every stage.
- **Bills that skipped Lords committee are excluded:** 111 bills had "Committee negatived" or
  "Order of Commitment discharged" in the Lords, so they have no Lords committee segment.
- **Correlation is not causation:** these are associations only, and are worth testing as
  features in the passage model.
