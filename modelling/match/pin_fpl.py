"""Write modelling/requirements-fpl.txt: exact versions of what the FPL workflow installs.

python -m modelling.match.pin_fpl
"""

from importlib.metadata import PackageNotFoundError, version

from data.common.config import ROOT

# What the FPL commands import, directly or through P1's code.
PACKAGES = ["numpy", "scipy", "pandas", "requests", "python-dotenv", "SQLAlchemy", "psycopg"]


def main() -> None:
    lines = []
    for name in PACKAGES:
        try:
            lines.append(f"{name}=={version(name)}\n")
        except PackageNotFoundError:
            print(f"Not installed here, so not pinned: {name}")
    path = ROOT / "modelling" / "requirements-fpl.txt"
    path.write_text("".join(lines), encoding="utf-8")
    print("".join(lines), end="")
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
