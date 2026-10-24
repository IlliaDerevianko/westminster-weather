"""Bills API stage IDs → model states (MODEL_SPEC §2.1, DATA_CONTRACT §2.5).

This is the Python twin of the ``stage_map`` reference table, so the API-backed interface and
a future database implementation share one mapping.
"""

from westminster_weather.domain.stages import BillStage

STAGE_ID_TO_STATE: dict[int, BillStage] = {
    6: BillStage.COMMONS_FIRST_READING,
    # 2nd reading absorbs programme motion (14), money resolution (15), ways and means (36)
    # and second reading committee (25).
    7: BillStage.COMMONS_SECOND_READING,
    14: BillStage.COMMONS_SECOND_READING,
    15: BillStage.COMMONS_SECOND_READING,
    25: BillStage.COMMONS_SECOND_READING,
    36: BillStage.COMMONS_SECOND_READING,
    8: BillStage.COMMONS_COMMITTEE,  # Public Bill Committee
    43: BillStage.COMMONS_COMMITTEE,
    48: BillStage.COMMONS_COMMITTEE,
    49: BillStage.COMMONS_COMMITTEE,  # Committee of the whole House
    9: BillStage.COMMONS_REPORT_STAGE,
    10: BillStage.COMMONS_THIRD_READING,
    1: BillStage.LORDS_FIRST_AND_SECOND_READING,
    2: BillStage.LORDS_FIRST_AND_SECOND_READING,
    30: BillStage.LORDS_FIRST_AND_SECOND_READING,
    3: BillStage.LORDS_COMMITTEE,
    46: BillStage.LORDS_COMMITTEE,
    4: BillStage.LORDS_REPORT_AND_THIRD_READING,
    5: BillStage.LORDS_REPORT_AND_THIRD_READING,
    12: BillStage.PING_PONG,
    13: BillStage.PING_PONG,
    42: BillStage.PING_PONG,
    11: BillStage.ROYAL_ASSENT,
}

CARRY_OVER_STAGE_IDS: tuple[int, ...] = (18, 19)
COMMITTEE_OF_WHOLE_HOUSE_STAGE_ID = 49

_COMMONS = (
    BillStage.COMMONS_FIRST_READING,
    BillStage.COMMONS_SECOND_READING,
    BillStage.COMMONS_COMMITTEE,
    BillStage.COMMONS_REPORT_STAGE,
    BillStage.COMMONS_THIRD_READING,
)
_LORDS = (
    BillStage.LORDS_FIRST_AND_SECOND_READING,
    BillStage.LORDS_COMMITTEE,
    BillStage.LORDS_REPORT_AND_THIRD_READING,
)
_END = (BillStage.PING_PONG, BillStage.ROYAL_ASSENT)

# States that may legitimately never be entered (MODEL_SPEC §2.2).
OPTIONAL_STATES: frozenset[BillStage] = frozenset({BillStage.PING_PONG})


def chain(originating_house: str) -> list[BillStage]:
    """Ordered model states for a bill (MODEL_SPEC §2.2)."""
    first, second = (_LORDS, _COMMONS) if originating_house == "Lords" else (_COMMONS, _LORDS)
    return [*first, *second, *_END]


def state_house(state: BillStage) -> str | None:
    """House whose sitting days count time in ``state`` (None = either House)."""
    if state in _COMMONS:
        return "Commons"
    if state in _LORDS:
        return "Lords"
    return None
