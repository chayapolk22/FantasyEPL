"""
Downloads past seasons' gameweek-by-gameweek data.

The official FPL API only serves the current season's match history, so past
seasons come from the community archive github.com/vaastav/Fantasy-Premier-League,
which snapshots the same API every gameweek (same columns as element-summary).

Only 2025-26 is pulled by default: that season introduced defensive
contribution points, so earlier seasons were scored under different rules —
a structural break, like mixing returns from before and after a regime change.
"""

import argparse
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

ARCHIVE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{path}"

# merged_gw: every player's every match. players_raw: maps each season's
# player id to FPL's stable `code`. fixtures: difficulty ratings.
FILES = ["gws/merged_gw.csv", "players_raw.csv", "fixtures.csv", "teams.csv"]

DEFAULT_SEASONS = ["2025-26"]


def fetch_season(season: str) -> None:
    out_dir = RAW_DIR / season
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in FILES:
        url = ARCHIVE_URL.format(season=season, path=path)
        print(f"Fetching {url} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        out_path = out_dir / Path(path).name
        out_path.write_bytes(response.content)
        print(f"Saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seasons", nargs="*", default=DEFAULT_SEASONS,
                        help="seasons to fetch, e.g. 2025-26")
    args = parser.parse_args()
    for season in args.seasons:
        fetch_season(season)


if __name__ == "__main__":
    main()
