# Stage 1 pitch: "On day one nobody can tell. Watch the repetition."

**The question.** A minister says "we will legislate on X". Will a Government Bill on X actually be introduced, and when?

**What we built.** From 25,242 written ministerial statements and Hansard speeches (2006–2026) we found 1,939 ministers' commitments to legislate, grouped them into 985 promises, and linked 192 of them to the Government Bill that delivered them. A hand-check of 150 decisions found them all correct. A weekly hazard model then forecasts each promise as time passes, using only what was known on the day of the forecast.

**Finding 1: on day one, nobody can tell.** At the moment a promise is made, our model is no better than the average rate (bill within a year: AUC 0.55, Brier 0.108 vs 0.107 for the average). A single sentence does not reveal which promises will be kept.

**Finding 2: repetition is the signal.** One year after the promise, among the 708 promises still waiting:

| Repeated in the first year | Promises | Bill in the next year (real) | Model predicted |
|---|---|---|---|
| Never | 628 | 2.1% | 2.2% |
| 1–2 times | 65 | 4.6% | 7.9% |
| 3 or more times | 15 | **20.0%** | 18.9% |

Promises that ministers keep repeating are about ten times more likely to become a bill, and the model's forecasts match those levels.

**Real examples** (model trained only on earlier promises):
- Post Office compensation, promised Sep 2022: 20% on day one; **95%** a year later after six repeats. Bill introduced Nov 2023.
- Leasehold reform, promised May 2018: 29% → 48% as mentions grew. Bill introduced May 2021, inside the forecast window.
- Guardianship of missing persons, promised Mar 2015, mentioned once more: 15% → 3%. The government never introduced a bill.

**Honest limits.** Small numbers in the most-repeated group (15 promises); the year-later figures are measured on the training data; Government Bills only, so a promise delivered through a backed Private Member's Bill counts as not taken up; a policy can be repeated for years and still dropped (conversion practices).

Reproduce: `uv run python -m westminster_weather.takeup.insample` (day one) and `--updating` (one year later), `uv run python -m westminster_weather.takeup.replay` (examples).
