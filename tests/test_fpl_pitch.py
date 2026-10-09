import xml.etree.ElementTree as ET

from modelling.match.fpl_pitch import headline, svg

POSITIONS = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
STARTS = [True, False] + [True] * 4 + [False] + [True] * 4 + [False] + [True] * 2 + [False]


def report(shown="transfers") -> dict:
    players = [
        {
            "id": i,
            "name": f"Player {i}",
            "club": "ARS",
            "position": pos,
            "price": 5.0,
            "next_points": 15 - i,
            "horizon_points": 50.0,
            "start": start,
            "captain": i == 2,
            "vice": i == 3,
            "bench_order": 0,
            "transfer_in": i == 7,
            "opponents": ["LEE (H)"],
        }
        for i, (pos, start) in enumerate(zip(POSITIONS, STARTS, strict=True))
    ]
    for k, p in enumerate(p for p in players if not p["start"]):
        p["bench_order"] = k + 1
    plan = {
        "next_gw_points": 55.0,
        "hits": 0,
        "transfers_out": ["Old"],
        "transfers_in": ["Player 7"],
        "players": players,
    }
    return {
        "gameweek": 6,
        "deadline": "2026-10-10T10:00:00Z",
        "generated_at": "2026-10-09",
        "gameweeks": [6, 7],
        "shown": shown,
        "free_transfers": 1,
        "free_transfers_source": "estimate",
        "chips": None,
        "plans": {shown: plan},
    }


def test_picture_is_valid_svg_with_every_player_once():
    picture = svg(report(), "transfers")
    root = ET.fromstring(picture)  # raises an error if the XML is broken
    texts = [t.text for t in root.iter("{http://www.w3.org/2000/svg}text")]
    for i in range(15):
        assert texts.count(f"Player {i}") == 1
    assert texts.count("C") == 1 and texts.count("V") == 1 and texts.count("IN") == 1


def test_headline_names_the_transfer_and_the_free_transfer_source():
    lines = headline(report(), report()["plans"]["transfers"])
    assert lines[0] == "Transfers: 1 free (estimate)"
    assert lines[1] == "Old → Player 7"
    assert lines[-1] == "Chip advice needs your team ID in config.toml"
