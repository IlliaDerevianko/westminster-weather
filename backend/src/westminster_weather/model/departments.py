"""Merge renamed or abolished government departments into their current successor.

The Bills API gives the sponsoring department under its name at the time, so the same
department appears under several names (e.g. Department of Health → Department of Health
and Social Care). The model groups bills by the current name. The database keeps the
original names; this mapping is applied on the model side only.

Where a department was split, it maps to the one successor that took its main legislative
work (noted below). Names not listed are already current or have no successor.
"""

from __future__ import annotations

CURRENT_NAME: dict[str, str] = {
    # Health
    "Department of Health": "Department of Health and Social Care",
    # Housing and local government
    "Department for Communities and Local Government": (
        "Ministry of Housing, Communities and Local Government"
    ),
    "Department for Levelling Up, Housing and Communities": (
        "Ministry of Housing, Communities and Local Government"
    ),
    # Business and trade. BEIS was split in 2023 into Business and Trade, Energy Security
    # and Net Zero, and Science, Innovation and Technology; its bills map to Business and
    # Trade, the main successor for business legislation.
    "Department of Trade and Industry": "Department for Business and Trade",
    "Department for Business, Enterprise and Regulatory Reform": (
        "Department for Business and Trade"
    ),
    "Department for Innovation, Universities and Skills": "Department for Business and Trade",
    "Department for Business, Innovation and Skills": "Department for Business and Trade",
    "Department for Business, Energy and Industrial Strategy": (
        "Department for Business and Trade"
    ),
    "Department for International Trade": "Department for Business and Trade",
    # Energy
    "Department of Energy and Climate Change": "Department for Energy Security and Net Zero",
    # Education
    "Department for Education and Skills": "Department for Education",
    "Department for Children, Schools and Families": "Department for Education",
    # Culture. The digital brief moved to Science, Innovation and Technology in 2023.
    "Department for Digital, Culture, Media and Sport": "Department for Culture, Media and Sport",
    # Foreign affairs and development (merged in 2020)
    "Foreign and Commonwealth Office": "Foreign, Commonwealth and Development Office",
    "Department for International Development": "Foreign, Commonwealth and Development Office",
    # Brexit department closed in 2020; its remaining work went to the Cabinet Office
    "Department for Exiting the European Union": "Cabinet Office",
    # Territorial offices renamed in 2018
    "Wales Office": "Office of the Secretary of State for Wales",
    "Scotland Office": "Office of the Secretary of State for Scotland",
}


def current_department(name: str | None) -> str | None:
    """The current name of ``name``'s department (unchanged if not in the mapping)."""
    if name is None:
        return None
    name = " ".join(name.split())  # some API names carry stray whitespace
    seen = set()
    while name in CURRENT_NAME and name not in seen:
        seen.add(name)
        name = CURRENT_NAME[name]
    return name
