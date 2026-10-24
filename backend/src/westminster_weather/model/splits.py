"""Time-based train/test split (MODEL_SPEC §7). Never split randomly."""

from __future__ import annotations

from datetime import date

# Bills introduced in sessions 20-30 (2006-07 to 2017-19) train the model.
TRAIN_SESSIONS: tuple[int, ...] = tuple(range(20, 31))
# Bills introduced in sessions 34-39 (2019 onwards) are scored. The current session (40)
# is still in progress and is never scored.
TEST_SESSIONS: tuple[int, ...] = (34, 35, 36, 37, 38, 39)

# Training data is read as of the day before the first test session began (2019-10-15), so
# nothing the model learns from was known only after test forecasts start.
TRAIN_CUTOFF = date(2019, 10, 14)
