"""
Pulls core FPL data from the official public API and saves it locally.
No API key needed — this endpoint is open.
"""

import json
import requests
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


def fetch_and_save(url: str, filename: str) -> None:
    print(f"Fetching {url} ...")
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()

    out_path = DATA_DIR / filename
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved to {out_path}")


def main():
    # bootstrap-static contains: all players, all teams, gameweek metadata
    fetch_and_save(BOOTSTRAP_URL, "bootstrap_static.json")

    # fixtures contains: the full season fixture list with difficulty ratings
    fetch_and_save(FIXTURES_URL, "fixtures.json")


if __name__ == "__main__":
    main()
