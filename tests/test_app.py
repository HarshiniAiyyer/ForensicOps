from app import clean_numeric, load_github_data


def test_sanity():
    assert True


def test_clean_numeric_basic():
    assert clean_numeric("$1,000") == "1000"
    assert clean_numeric("25%") == "25"
    assert clean_numeric("$0") == "0"
    assert clean_numeric(42) == 42  # Already numeric, should stay the same


def test_load_github_data():
    url = "https://raw.githubusercontent.com/HarshiniAiyyer/Financial-Forensics/main/data/states.csv"
    df = load_github_data(url)
    assert df is not None
    assert "State Medicaid Expansion (2016)" in df.columns
