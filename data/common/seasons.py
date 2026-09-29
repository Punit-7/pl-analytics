from datetime import date


def current_season_start(today: date | None = None, start_month: int = 8) -> int:
    """Start year of the season in progress.
    28 Sep 2026 -> 2026. 10 Mar 2027 -> 2026. 31 Jul 2026 -> 2025."""
    today = today or date.today()
    return today.year if today.month >= start_month else today.year - 1


def season_code(start_year: int) -> str:
    """football-data.co.uk folder name. 1993 -> '9394', 1999 -> '9900'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def start_year_from_code(code: str) -> int:
    """'9394' -> 1993, '2627' -> 2026. Valid for seasons 1993 to 2092."""
    y = int(code[:2])
    return y + 1900 if y >= 93 else y + 2000


def season_label(start_year: int) -> str:
    """Display label. 2026 -> '2026/27', 1999 -> '1999/00'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"