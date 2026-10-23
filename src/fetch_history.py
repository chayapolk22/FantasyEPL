"""
Pulls per-player match history from the FPL API and saves the raw responses.

bootstrap-static only has season totals. element-summary/{id} has one row
per match played this season — the "return series" every model downstream
is built on.

Run fetch_data.py first: the player list comes from bootstrap_static.json.
"""

import argparse
import json
import time
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw" / "element_summary"

ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"

# Be polite to a free public API: ~5 requests/second, a few retries on errors.
REQUEST_DELAY_S = 0.2
MAX_RETRIES = 3


def load_player_ids() -> list[int]:
    with open(DATA_DIR / "bootstrap_static.json") as f:
        bootstrap = json.load(f)
    return [p["id"] for p in bootstrap["elements"]]


def fetch_one(session: requests.Session, player_id: int) -> dict:
    url = ELEMENT_SUMMARY_URL.format(player_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            print(f"  player {player_id}: {e} — retrying in {wait}s")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download players that already have a saved file",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    player_ids = load_player_ids()
    print(f"{len(player_ids)} players in bootstrap_static.json")

    session = requests.Session()
    fetched = skipped = 0
    for i, player_id in enumerate(player_ids, start=1):
        out_path = RAW_DIR / f"{player_id}.json"
        if out_path.exists() and not args.refresh:
            skipped += 1
            continue

        data = fetch_one(session, player_id)
        with open(out_path, "w") as f:
            json.dump(data, f)
        fetched += 1

        if i % 100 == 0:
            print(f"  {i}/{len(player_ids)}")
        time.sleep(REQUEST_DELAY_S)

    print(f"Done: {fetched} fetched, {skipped} already saved, in {RAW_DIR}")


if __name__ == "__main__":
    main()
