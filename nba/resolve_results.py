"""
Resolve actual game outcomes for any pending prediction files.

Usage:
    python resolve_results.py               # resolves all unresolved dates
    python resolve_results.py 2026-03-25    # resolves a specific date

For each date with an unresolved predictions file, fetches actual game results
from the NBA API and fills in ACTUAL_OUTCOME + CORRECT_PREDICTION.

If multiple prediction files exist for the same date (from re-runs), only the
latest one is resolved.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

from predict_games import update_actual_outcomes


PREDICTIONS_DIR = Path('predictions')


def find_pending_files() -> dict[str, Path]:
    """
    Return a dict of {date_str: latest_csv_path} for any prediction files
    that still have blank ACTUAL_OUTCOME values.
    """
    pending: dict[str, Path] = {}

    for csv_file in sorted(PREDICTIONS_DIR.glob('predictions_????-??-??.csv')):
        try:
            df = pd.read_csv(csv_file)
        except Exception:
            continue

        if 'ACTUAL_OUTCOME' not in df.columns:
            continue

        if df['ACTUAL_OUTCOME'].fillna('').eq('').any():
            # Extract the date portion: predictions_YYYY-MM-DD.csv
            date_str = csv_file.stem.split('_', 1)[1]  # 'YYYY-MM-DD'
            pending[date_str] = csv_file

    return pending


def resolve_date(date_str: str, csv_file: Path) -> bool:
    """Fetch results for date_str and update csv_file. Returns True on success."""
    # Convert YYYY-MM-DD -> MM/DD/YYYY for the NBA API
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    game_date = dt.strftime('%m/%d/%Y')

    print(f"\n{'='*60}")
    print(f"Resolving {date_str}  ({csv_file.name})")
    print(f"{'='*60}")

    result = update_actual_outcomes(str(csv_file), game_date=game_date)

    if result is None:
        print(f"  ✗ Failed to resolve {date_str}")
        return False

    completed = result[result['ACTUAL_OUTCOME'].fillna('') != '']
    if len(completed) == 0:
        print(f"  ✗ No completed games found for {date_str} — games may still be in progress.")
        return False

    accuracy = completed['CORRECT_PREDICTION'].mean()
    print(f"  ✓ {len(completed)} games resolved  |  accuracy: {accuracy:.1%}")
    return True


def main():
    if len(sys.argv) > 1:
        # Specific date provided
        date_str = sys.argv[1]
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            print(f"Invalid date format: {date_str}  (expected YYYY-MM-DD)")
            sys.exit(1)

        csv_file = PREDICTIONS_DIR / f'predictions_{date_str}.csv'
        if not csv_file.exists():
            print(f"No predictions file found for {date_str}")
            sys.exit(1)
        resolve_date(date_str, csv_file)

    else:
        # Scan for all unresolved files
        pending = find_pending_files()

        if not pending:
            print("No pending prediction files found — all dates are resolved.")
            return

        today = datetime.now().strftime('%Y-%m-%d')
        # Skip today: games haven't finished yet
        pending = {d: f for d, f in pending.items() if d < today}

        if not pending:
            print("No past dates with unresolved predictions.")
            return

        print(f"Found {len(pending)} unresolved date(s): {', '.join(sorted(pending))}")

        resolved = 0
        for date_str in sorted(pending):
            if resolve_date(date_str, pending[date_str]):
                resolved += 1

        print(f"\n{'='*60}")
        print(f"Done: {resolved}/{len(pending)} dates resolved")


if __name__ == '__main__':
    main()
