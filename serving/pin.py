"""Write the two requirements files with the exact versions installed in this environment.

python -m serving.pin
"""

from importlib.metadata import version

from data.common.config import ROOT

# What the API container needs, and what the weekly job and the tests need on top.
API = ["fastapi", "uvicorn", "numpy", "scipy", "pandas", "scikit-learn", "joblib", "SQLAlchemy"]
JOB = ["requests", "python-dotenv", "mlflow", "evidently", "httpx", "pytest", "ruff"]


def lines(packages: list[str]) -> str:
    return "".join(f"{name}=={version(name)}\n" for name in packages)


def main() -> None:
    folder = ROOT / "serving"
    (folder / "requirements.txt").write_text(lines(API), encoding="utf-8")
    (folder / "requirements-job.txt").write_text(
        "-r requirements.txt\n" + lines(JOB), encoding="utf-8"
    )
    print(lines(API + JOB))


if __name__ == "__main__":
    main()