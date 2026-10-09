"""Draw the suggested FPL team on a pitch: an SVG picture and an HTML page around it.

python -m modelling.match.fpl_pitch                  the plan the optimiser chose to show
python -m modelling.match.fpl_pitch --plan wildcard  another plan from fpl_plan.json
"""

import argparse
import json
from datetime import datetime
from html import escape

from data.common.config import ROOT

REPORTS = ROOT / "modelling" / "reports"
ROWS = ["GKP", "DEF", "MID", "FWD"]
W = 720  # picture width in pixels
CARD_W, CARD_H, GAP = 112, 86, 12
ROW_STEP = 112  # distance between the rows of the formation
FONT = "Segoe UI, Arial, sans-serif"
CHIPS = {
    "wildcard": "Wildcard",
    "freehit": "Free Hit",
    "bboost": "Bench Boost",
    "3xc": "Triple Captain",
}


def text(x, y, value, size=12, weight="normal", fill="#1b1024", anchor="middle") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(str(value))}</text>'
    )


def short(name: str, limit: int = 13) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "."


def card(x: float, y: float, p: dict, label: str | None = None) -> str:
    """One player: name, next opponent(s), projected points and price, with C, V and IN marks."""
    cx = x + CARD_W / 2
    parts = [
        f'<rect x="{x}" y="{y}" width="{CARD_W}" height="{CARD_H}" rx="8" fill="#ffffff" '
        f'stroke="{"#e8590c" if p["transfer_in"] else "#d0c8d8"}" '
        f'stroke-width="{3 if p["transfer_in"] else 1}"/>',
        f'<rect x="{x}" y="{y}" width="{CARD_W}" height="22" rx="8" fill="#37003c"/>',
        f'<rect x="{x}" y="{y + 14}" width="{CARD_W}" height="8" fill="#37003c"/>',
        text(cx, y + 16, label or f"{p['club']} · {p['position']}", 11, "bold", "#ffffff"),
        text(cx, y + 40, short(p["name"]), 13, "bold"),
        text(cx, y + 57, ", ".join(p["opponents"]) or "No match", 11, fill="#5a4d66"),
        text(cx, y + 75, f"{p['next_points']:.1f} pts · £{p['price']:.1f}m", 11),
    ]
    badge = "C" if p["captain"] else "V" if p["vice"] else None
    if badge:
        parts += [
            f'<circle cx="{x + 4}" cy="{y + 4}" r="11" fill="#1b1024" stroke="#ffffff" '
            'stroke-width="2"/>',
            text(x + 4, y + 8, badge, 12, "bold", "#ffffff"),
        ]
    if p["transfer_in"]:
        parts += [
            f'<rect x="{x + CARD_W - 30}" y="{y - 9}" width="34" height="18" rx="4" '
            'fill="#e8590c"/>',
            text(x + CARD_W - 13, y + 4, "IN", 11, "bold", "#ffffff"),
        ]
    return "\n".join(parts)


def row_x(n: int) -> list[float]:
    """Left edges of n cards centred across the picture."""
    total = n * CARD_W + (n - 1) * GAP
    return [(W - total) / 2 + i * (CARD_W + GAP) for i in range(n)]


def headline(report: dict, plan: dict) -> list[str]:
    """Lines of text above the pitch: the transfers (three per line), then the chip advice."""
    if report["shown"] == "transfers":
        hits = plan["hits"]
        lines = [
            f"Transfers: {report['free_transfers']} free ({report['free_transfers_source']})"
            + (f", {hits} paid = −{4 * hits} points" if hits else "")
        ]
        moves = [
            f"{o} → {i}" for o, i in zip(plan["transfers_out"], plan["transfers_in"], strict=True)
        ]
        if not moves:
            lines.append("No transfer: save the free transfer")
        lines += ["   ·   ".join(moves[k : k + 3]) for k in range(0, len(moves), 3)]
    else:
        lines = [
            {
                "wildcard": "A fresh 15 for the next gameweeks (Wildcard)",
                "freehit": "A fresh 15 for one gameweek (Free Hit)",
            }[report["shown"]]
        ]
    chips = report.get("chips")
    if not chips:
        lines.append("Chip advice needs your team ID in config.toml")
    elif chips["play"]:
        c = next(c for c in chips["checks"] if c["chip"] == chips["play"])
        lines.append(f"Chip: play {c['name']} (+{c['gain']:.1f} projected; limit {c['limit']:g})")
    else:
        lines.append("Chip: none this week (no chip passes its limit)")
    return lines


def svg(report: dict, plan_name: str) -> str:
    plan = report["plans"][plan_name]
    players = plan["players"]
    deadline = datetime.fromisoformat(report["deadline"].replace("Z", "+00:00"))
    lines = headline(report, plan)
    pitch_top = 110 + 20 * len(lines)  # the pitch starts below the text
    bench_top = pitch_top + 4 * ROW_STEP + 20
    height = bench_top + CARD_H + 76
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" '
        f'viewBox="0 0 {W} {height}" role="img" aria-label="Suggested FPL team">',
        f'<rect width="{W}" height="{height}" fill="#f6f3f9"/>',
        text(W / 2, 34, f"Gameweek {report['gameweek']} — suggested team", 22, "bold"),
        text(
            W / 2,
            58,
            f"Deadline {deadline:%a %d %b, %H:%M} UTC · projected "
            f"{plan['next_gw_points']:.1f} points · made {report['generated_at'][:10]}",
            12,
            fill="#5a4d66",
        ),
    ]
    for i, line in enumerate(lines):
        out.append(text(W / 2, 86 + i * 20, line, 13, "bold" if i == 0 else "normal"))
    top, bottom = pitch_top - 14, pitch_top + 4 * ROW_STEP - 10
    out += [
        f'<rect x="16" y="{top}" width="{W - 32}" height="{bottom - top}" rx="10" fill="#1f8a4c"/>',
        f'<rect x="40" y="{top + 10}" width="{W - 80}" height="{bottom - top - 20}" fill="none" '
        'stroke="#ffffff" stroke-opacity="0.5" stroke-width="2"/>',
        f'<line x1="40" y1="{(top + bottom) / 2}" x2="{W - 40}" y2="{(top + bottom) / 2}" '
        'stroke="#ffffff" stroke-opacity="0.5" stroke-width="2"/>',
        f'<circle cx="{W / 2}" cy="{(top + bottom) / 2}" r="56" fill="none" stroke="#ffffff" '
        'stroke-opacity="0.5" stroke-width="2"/>',
    ]
    starters = [p for p in players if p["start"]]
    for r, position in enumerate(ROWS):
        line = [p for p in starters if p["position"] == position]
        line.sort(key=lambda p: -p["next_points"])
        for x, p in zip(row_x(len(line)), line, strict=True):
            out.append(card(x, pitch_top + r * ROW_STEP, p))
    bench = sorted((p for p in players if not p["start"]), key=lambda p: p["bench_order"])
    out += [
        f'<rect x="16" y="{bench_top - 16}" width="{W - 32}" height="{CARD_H + 32}" rx="10" '
        'fill="#e4dcec"/>',
    ]
    for x, p in zip(row_x(len(bench)), bench, strict=True):
        label = f"{p['bench_order']}. {p['club']} · {p['position']}"
        out.append(card(x, bench_top, p, label))
    foot = bench_top + CARD_H + 40
    out += [
        text(
            W / 2,
            foot,
            "Projection: pl-analytics Dixon–Coles model. Data: Fantasy Premier "
            "League API, football-data.co.uk.",
            11,
            fill="#5a4d66",
        ),
        text(
            W / 2,
            foot + 16,
            "A model suggestion, not advice. Not affiliated with the Premier League or FPL.",
            11,
            fill="#5a4d66",
        ),
        "</svg>",
    ]
    return "\n".join(out)


def html_page(report: dict, plan_name: str, picture: str) -> str:
    """The picture, then the three plans side by side and the chip checks."""
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{p['next_gw_points']:.1f}</td>"
        f"<td>{p['horizon_value']:.1f}</td><td>{p['hits']}</td></tr>"
        for name, p in report["plans"].items()
    )
    chips = report.get("chips") or {"checks": []}
    checks = (
        "".join(
            f"<tr><td>{escape(c['name'])}</td><td>{c['gain']:.1f}</td><td>{c['limit']:g}</td>"
            f"<td>{'yes' if c['passes'] else 'no'}</td></tr>"
            for c in chips["checks"]
        )
        or "<tr><td colspan='4'>No team set, or no chip available</td></tr>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FPL gameweek {report["gameweek"]}</title>
<style>
body {{ margin: 0; background: #f6f3f9; color: #1b1024; font-family: {FONT}; }}
main {{ max-width: 760px; margin: 0 auto; padding: 16px; }}
svg {{ width: 100%; height: auto; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0 24px; }}
th, td {{ border: 1px solid #d0c8d8; padding: 6px 8px; text-align: left; }}
th {{ background: #e4dcec; }}
</style></head>
<body><main>
{picture}
<h2>The three plans</h2>
<p>Shown above: <strong>{escape(plan_name)}</strong>. Horizon value: projected points over
gameweeks {report["gameweeks"][0]}–{report["gameweeks"][-1]}, after transfer hits.</p>
<table><tr><th>Plan</th><th>Next gameweek</th><th>Horizon value</th><th>Hits</th></tr>{rows}</table>
<h2>Chip checks</h2>
<table><tr><th>Chip</th><th>Projected gain</th><th>Limit</th><th>Passes</th></tr>{checks}</table>
</main></body></html>
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Draw the suggested FPL team")
    parser.add_argument("--plan", help="transfers, wildcard or freehit; default: the shown plan")
    args = parser.parse_args(argv)
    report = json.loads((REPORTS / "fpl_plan.json").read_text(encoding="utf-8"))
    name = args.plan or report["shown"]
    if name not in report["plans"]:
        raise SystemExit(f"No {name} plan in fpl_plan.json; plans: {list(report['plans'])}")
    if name != report["shown"]:
        report = {**report, "shown": name}
    picture = svg(report, name)
    (REPORTS / "fpl_pitch.svg").write_text(picture, encoding="utf-8")
    (REPORTS / "fpl_pitch.html").write_text(html_page(report, name, picture), encoding="utf-8")
    print(f"Wrote {REPORTS / 'fpl_pitch.svg'} and fpl_pitch.html")


if __name__ == "__main__":
    main()
