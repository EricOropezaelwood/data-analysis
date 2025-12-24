from nba_api.stats.endpoints import leaguegamelog
import pandas as pd
from data_cleaning import clean_data
from preprocessing import create_pregame_features
from xgboost_analysis import find_top_features
from save_results import save_test_results_to_csv
from get_injuries import get_single_game_injuries, get_season_game_injuries
import pickle
from pathlib import Path
from datetime import datetime


# from nba_api
def get_league_game_log(season, use_cache=True, force_refresh=False):
    cache_file = Path(f"game_log_{season}.pkl")
    cached_data = None

    # Load existing cache if available (rate limiting...)
    if use_cache and cache_file.exists() and not force_refresh:
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)

        # check if game(s) are cached
        if "GAME_DATE" in cached_data.columns:
            latest_date = pd.to_datetime(cached_data["GAME_DATE"]).max()
            total_cached_games = len(cached_data)
            print(
                f"  Cached: {total_cached_games} games (latest: {latest_date.strftime('%Y-%m-%d')})"
            )

            # Check if we might need new data (cache is older than today)
            if latest_date.date() < datetime.now().date():
                print(f"  Cache may be outdated, checking for new games...")
            else:
                print(f"  Cache is up to date")
                return cached_data
        # game(s) are not cached
        else:
            print(f"  Cached: {len(cached_data)} games")
            return cached_data

    print(f"Fetching game log for {season} season from NBA API...")

    try:
        game_log = leaguegamelog.LeagueGameLog(
            season=season,
            timeout=120,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept": "application/json",
                "Referer": "https://www.stats.nba.com/",
            },
        )
        new_data = game_log.get_data_frames()[0]

        # If we have cached data, merge it with new data
        if cached_data is not None and "GAME_DATE" in new_data.columns:
            # Convert dates for comparison
            new_data["GAME_DATE"] = pd.to_datetime(new_data["GAME_DATE"])
            cached_data["GAME_DATE"] = pd.to_datetime(cached_data["GAME_DATE"])

            # Find games not in cache (by GAME_ID if available, otherwise by date)
            if "GAME_ID" in new_data.columns:
                existing_game_ids = set(cached_data["GAME_ID"])
                new_games = new_data[~new_data["GAME_ID"].isin(existing_game_ids)]
            else:
                latest_cached_date = cached_data["GAME_DATE"].max()
                new_games = new_data[new_data["GAME_DATE"] > latest_cached_date]

            if len(new_games) > 0:
                print(f"  Found {len(new_games)} new games since cache")
                df = pd.concat([cached_data, new_games], ignore_index=True)
                df = df.sort_values("GAME_DATE").reset_index(drop=True)
            else:
                print(f"  No new games found, using cached data")
                df = cached_data
        else:
            df = new_data

        # Save updated cache
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
        print(f"Data cached to {cache_file} ({len(df)} total games)")

        return df

    except Exception as e:
        print(f"Attempt failed: {e}")
        if cached_data is not None:
            print(f"Using existing cached data")
            return cached_data
        elif cache_file.exists():
            print(f"Falling back to cached data from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        else:
            raise


def get_multiple_seasons(seasons, use_cache=True, force_refresh=False):

    if isinstance(seasons, (str, int)):
        seasons = [seasons]

    print(
        f"\nFetching data for {len(seasons)} season(s): {', '.join(map(str, seasons))}"
    )
    print("=" * 60)

    all_data = []

    for season in seasons:
        try:
            df = get_league_game_log(
                season, use_cache=use_cache, force_refresh=force_refresh
            )
            df["SEASON"] = str(season)
            all_data.append(df)
            print(f"✓ Successfully loaded {season}: {len(df)} games")
        except Exception as e:
            print(f"✗ Failed to load {season}: {e}")

    if not all_data:
        raise ValueError("No data loaded for any season")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n{'=' * 60}")
    print(f"Total games across all seasons: {len(combined_df)}")
    print(f"Seasons included: {sorted(combined_df['SEASON'].unique())}")
    print(f"{'=' * 60}\n")

    return combined_df


if __name__ == "__main__":
    # ========== CONFIGURE SEASONS HERE ==========
    # Option 1: Single season
    # SEASONS = 2025

    # Option 2: Multiple seasons
    SEASONS = [2023, 2024, 2025]

    # Option 3: Season with format 'YYYY-YY'
    # SEASONS = ['2022-23', '2023-24', '2024-25']
    # ============================================

    # Get the game log data
    game_log = get_multiple_seasons(SEASONS)
    print(game_log)

    # get injuries
    injuries_log = get_season_game_injuries(game_log)

    print(injuries_log)

    # # Clean the data
    # cleaned_data = clean_data(game_log, target_col='WL')

    # # Create pre-game features (rolling averages, season stats, etc.)
    # # This ensures we only use information available BEFORE each game
    # # Features are defined in features_config.json
    # cleaned_data = create_pregame_features(cleaned_data)

    # # Show date range in cleaned data
    # if 'GAME_DATE' in cleaned_data.columns:
    #     dates = pd.to_datetime(cleaned_data['GAME_DATE'])
    #     print(f"\n{'='*60}")
    #     print("CLEANED DATA DATE RANGE")
    #     print(f"{'='*60}")
    #     print(f"Total games: {len(cleaned_data)}")
    #     print(f"Earliest game: {dates.min().strftime('%Y-%m-%d')}")
    #     print(f"Latest game: {dates.max().strftime('%Y-%m-%d')}")
    #     print(f"Date span: {(dates.max() - dates.min()).days} days")

    #     # Show games by season
    #     if 'SEASON' in cleaned_data.columns:
    #         print(f"\nGames by season:")
    #         for season in sorted(cleaned_data['SEASON'].unique()):
    #             season_data = cleaned_data[cleaned_data['SEASON'] == season]
    #             season_dates = pd.to_datetime(season_data['GAME_DATE'])
    #             print(f"  {season}: {len(season_data)} games ({season_dates.min().strftime('%Y-%m-%d')} to {season_dates.max().strftime('%Y-%m-%d')})")
    #     print(f"{'='*60}\n")

    # # XGBoost analysis for feature importance
    # print("\n" + "="*60)
    # print("XGBoost Feature Importance Analysis")
    # print("="*60)

    # print("\n2. USING PRE-GAME FEATURES (no data leakage):")
    # top_features_no_pm, train_acc_no_pm, test_acc_no_pm, X_test, y_test, y_pred, y_pred_proba = find_top_features(
    #     cleaned_data,
    #     target_col='WL',
    #     top_n=20,
    #     exclude_features=['PLUS_MINUS', 'MIN']
    # )

    # print(f"Model Accuracy - Train: {train_acc_no_pm:.3f}, Test: {test_acc_no_pm:.3f}")
    # print("\nTop features by XGBoost gain:")
    # for feature, gain in top_features_no_pm.items():
    #     print(f"  {feature}: {gain:.2f}")

    # # Save test results to CSV (for me to explore)
    # csv_file = save_test_results_to_csv(
    #     X_test=X_test,
    #     y_test=y_test,
    #     y_pred=y_pred,
    #     y_pred_proba=y_pred_proba,
    #     train_acc=train_acc_no_pm,
    #     test_acc=test_acc_no_pm,
    #     original_data=cleaned_data,
    #     output_dir='test_results'
    # )
