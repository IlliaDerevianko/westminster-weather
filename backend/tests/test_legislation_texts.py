"""legislation.gov.uk Act XML parsing for `act_texts` and `act_in_force_status`."""

from datetime import date

from westminster_weather.clients.legislation_texts import parse_current, parse_enacted

NS = 'xmlns="http://www.legislation.gov.uk/namespaces/legislation"'

ENACTED = f"""<Legislation {NS} NumberOfProvisions="3"><Primary>
<PrimaryPrelims><Title>Example Act 2026</Title></PrimaryPrelims>
<Body><P1group><Title>Commencement</Title><P1><Text>This Act comes into force on the day
on which this Act is passed.</Text></P1></P1group></Body></Primary></Legislation>"""


def _current(statuses: list[str | None]) -> bytes:
    parts = []
    for i, s in enumerate(statuses):
        status = f' Status="{s}"' if s else ""
        parts.append(
            f'<P1group{status} RestrictStartDate="2026-0{i + 1}-01">'
            "<P1><Text>x</Text></P1></P1group>"
        )
    groups = "".join(parts)
    return f"<Legislation {NS}><Primary><Body>{groups}</Body></Primary></Legislation>".encode()


def test_enacted_text_is_plain_and_whitespace_normalised():
    out = parse_enacted(ENACTED.encode())
    assert out["text"] == (
        "Example Act 2026 Commencement This Act comes into force on the day on which this Act "
        "is passed."
    )
    assert out["n_characters"] == len(out["text"])
    assert out["number_of_provisions"] == 3


def test_act_with_prospective_sections_is_not_fully_in_force():
    out = parse_current(_current([None, "Prospective", "Prospective"]))
    assert out["n_section_groups"] == 3
    assert out["n_section_groups_prospective"] == 2
    assert not out["fully_in_force"]


def test_act_with_no_prospective_provision_is_fully_in_force_on_latest_start():
    out = parse_current(_current([None, None]))
    assert out["fully_in_force"]
    assert out["n_prospective_elements"] == 0
    assert out["latest_in_force_start_date"] == date(2026, 2, 1)
