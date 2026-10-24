from westminster_weather.model.departments import CURRENT_NAME, current_department


def test_renamed_departments_map_to_current_names():
    assert current_department("Department of Health") == "Department of Health and Social Care"
    assert (
        current_department("Department for Communities and Local Government")
        == "Ministry of Housing, Communities and Local Government"
    )
    assert (
        current_department("Department for Business, Innovation and Skills")
        == "Department for Business and Trade"
    )
    assert current_department("HM Treasury") == "HM Treasury"
    assert current_department("Department for Exiting the European Union ") == "Cabinet Office"
    assert current_department(None) is None


def test_every_target_is_a_current_name():
    assert not set(CURRENT_NAME.values()) & set(CURRENT_NAME)
