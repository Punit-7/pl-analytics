import pytest

from data.common.io_utils import DataValidationError, write_atomic
from data.ingest.football_data import validate

GOOD = (
    b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
    b"E0,16/08/2024,Man United,Fulham,1,0,H\n"
    b"E0,17/08/2024,Ipswich,Liverpool,0,2,A\n"
    b",,,,,,\n"
)  # a blank row, as in real files


def test_validate_counts_matches_and_ignores_blank_rows():
    assert validate(GOOD, "2425") == 2


def test_validate_rejects_missing_columns():
    bad = b"Div,Date,HomeTeam\nE0,16/08/2024,Man United\n"
    with pytest.raises(DataValidationError, match="missing columns"):
        validate(bad, "2425")


def test_validate_rejects_html_error_page():
    with pytest.raises(DataValidationError):
        validate(b"<html><body>Not found</body></html>", "2425")


def test_write_atomic_replaces_file_and_leaves_no_tmp(tmp_path):
    target = tmp_path / "x.csv"
    target.write_bytes(b"old")
    write_atomic(target, b"new")
    assert target.read_bytes() == b"new"
    assert list(tmp_path.glob("*.tmp")) == []
