# Progress by originating House

**Question:** does a UK Government Bill move differently through the Commons, the Lords and on
to Royal Assent depending on which House it starts in?

Regenerate everything in this folder from `backend/`:

```bash
uv run python -m westminster_weather.analysis.origin_comparison
```

## Data

- **Bills:** all 588 Government Bills: 438 start in the Commons and 150 in the Lords.
- **Time in a House:** from that House's 1st reading to its 3rd reading, in calendar days.
- **End-game:** from the second House's 3rd reading to Royal Assent, which covers ping-pong
  and waiting for Assent.
- **Total:** from the bill's introduction to Royal Assent.
- **Two groups:** *all bills*, and *regular bills*. Regular bills exclude those that cleared
  their first House in 14 days or less. **108 Commons-first bills are fast-tracked against
  only 1 Lords-first bill.** All 79 finance and supply-type bills must start in the Commons,
  so the all-bills comparison flatters the Commons.
- **Test:** Mann-Whitney U for time differences, Fisher's exact test for pass rates.

## Findings

Median calendar days:

| Part of the journey | Commons-first (all) | Lords-first (all) | Commons-first (regular) | Lords-first (regular) |
|---|---|---|---|---|
| First House | 74 | 110 | 98 | 110 * |
| Second House | 54 | 96 | 95 | 96 (n.s.) |
| End-game → Royal Assent | 8 | 23 | 14 | 23 ** |
| **Total to Royal Assent** | **141** | **252** | **217** | **254 \*** |

\* p<0.05, ** p<0.01, n.s. = not significant (regular-bills comparison).

1. **Lords-first bills take longer overall:** a median of 252 against 141 days. Most of that
   gap comes from the fast Commons-only finance and emergency bills. For regular bills the
   gap narrows to **254 against 217 days, about 5 weeks**, and is still statistically
   significant.
2. **The Lords is slower when it goes first:** a median of 110 days as the first House against
   95 days as the second House (p<0.001).
3. **The Commons takes the same time either way:** 98 days first, 96 days second (not
   significant).
4. **Lords-first bills wait longer at the end:** a median of 23 against 14 days from the
   second House's 3rd reading to Royal Assent (p<0.01). A likely reason is that Commons
   amendments send the bill back to the Lords for ping-pong. This chart does not check that
   directly.
5. **Pass rates are similar:** 92% of closed Commons-first bills and 89% of Lords-first bills
   reached Royal Assent. The difference is not significant (p = 0.22). Lords-first bills fail
   slightly more often at session end (5% against 2%) and at dissolution (6% against 4%).

## Charts

| File | What it shows |
|---|---|
| `1_journey_by_origin.png` | Box plots of each part of the journey, Commons-first (green) against Lords-first (red), for all bills (top) and regular bills (bottom). |
| `2_house_first_vs_second.png` | Median time in each House when it goes first against when it goes second (regular bills). |
| `3_outcomes_by_origin.png` | How closed bills end (Royal Assent, fell, withdrawn) by originating House. |

`durations_by_origin_per_bill.csv` has every bill's durations and its fast-tracked flag.
`metrics.json` has every median, count and p-value.

## Caveats

- **Different kinds of bill:** the government chooses which House a bill starts in.
  Lords-first bills tend to be technical or less politically urgent, so the differences may
  reflect the bill's type rather than the House it starts in.
- **Calendar days include recesses.**
- **Bill size and session timing are not controlled for:** `size_pages` and `size_clauses`
  are empty in the database.
- **Bills not yet finished are left out:** a bill only counts for a part of the journey it
  has finished, so in-progress and failed bills drop out of the later comparisons.
