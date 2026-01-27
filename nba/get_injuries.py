from nba_api.stats.endpoints import boxscoresummaryv3
import pandas as pd
import pickle
import time
import random


def get_single_game_injuries(game_id):
    """
    Get injuries for a single game.
    Mainly
    example call:
        injuries_log = get_single_game_injuries(
            game_log["GAME_ID"].iloc[0])
    """
    boxscore = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id)
    injuries = boxscore.get_data_frames()[
        5
    ]  # 5 is the index of the InactivePlayers dataframe
    return injuries


def get_season_game_injuries(
    gamelog, delay_range=(0.6, 1.2), max_retries=3, backoff_factor=2
):
    """
    Get injuries for all games across multiple seasons with rate limiting protection
    Each season is processed separately with its own cache file
    """
    # Get unique seasons
    seasons = sorted(gamelog["SEASON"].unique())

    print(f"\n{'='*60}")
    print(f"PROCESSING INJURY DATA FOR {len(seasons)} SEASON(S)")
    print(f"Seasons: {', '.join(map(str, seasons))}")
    print(f"{'='*60}")

    all_seasons_injuries = []

    # Process each season separately
    for season in seasons:
        season_gamelog = gamelog[gamelog["SEASON"] == season].reset_index(drop=True)

        # Cache file for this specific season
        cache_file = f"injuries_cache_{season}.pkl"

        # Try to load existing cache
        try:
            with open(cache_file, "rb") as f:
                season_injuries = pickle.load(f)
        except FileNotFoundError:
            season_injuries = pd.DataFrame()

        # Get list of games already processed
        processed_games = (
            set(season_injuries["gameId"].unique())
            if not season_injuries.empty
            else set()
        )

        # Log cache status with game indices
        total_games = len(season_gamelog["GAME_ID"].unique())
        cached_games = len(processed_games)
        remaining_games = total_games - cached_games

        # Find which game indices are cached
        cached_indices = []
        for idx, game_id in enumerate(season_gamelog["GAME_ID"], start=1):
            if game_id in processed_games:
                cached_indices.append(idx)

        print(f"\n{'='*60}")
        print(f"SEASON {season} - INJURY DATA CACHE STATUS")
        print(f"{'='*60}")
        print(f"Cache file: {cache_file}")
        print(f"Total games in season: {total_games}")
        print(f"Already cached: {cached_games}")

        # Show cached game ranges
        if cached_indices:
            ranges = []
            start = cached_indices[0]
            end = cached_indices[0]

            for i in range(1, len(cached_indices)):
                if cached_indices[i] == end + 1:
                    end = cached_indices[i]
                else:
                    if start == end:
                        ranges.append(f"#{start}")
                    else:
                        ranges.append(f"#{start}-{end}")
                    start = cached_indices[i]
                    end = cached_indices[i]

            # Add the last range
            if start == end:
                ranges.append(f"#{start}")
            else:
                ranges.append(f"#{start}-{end}")

            # Print ranges (limit to first 5 ranges to avoid clutter)
            if len(ranges) <= 5:
                print(f"Cached games: {', '.join(ranges)}")
            else:
                print(f"Cached games: {', '.join(ranges[:5])} ... (and more)")

        print(f"Remaining to fetch: {remaining_games}")
        print(
            f"Delay between requests: {delay_range[0]}-{delay_range[1]}s (randomized)"
        )
        print(f"{'='*60}\n")

        # Skip if all games are cached
        if remaining_games == 0:
            print(f"✓ Season {season}: All games already cached, skipping fetch\n")
            all_seasons_injuries.append(season_injuries)
            continue

        # Process each game
        games_processed = 0
        for idx, game_id in enumerate(season_gamelog["GAME_ID"], start=1):
            if game_id in processed_games:
                continue

            # Retry logic with exponential backoff
            for attempt in range(max_retries):
                try:
                    # Add delay before API call (except for first game)
                    if games_processed > 0:
                        delay = random.uniform(delay_range[0], delay_range[1])
                        time.sleep(delay)

                    injuries = get_single_game_injuries(game_id)
                    season_injuries = pd.concat(
                        [season_injuries, injuries], ignore_index=True
                    )
                    processed_games.add(game_id)
                    games_processed += 1

                    # Save after each game to preserve progress
                    with open(cache_file, "wb") as f:
                        pickle.dump(season_injuries, f)

                    # Progress indicator every 10 games
                    if games_processed % 10 == 0:
                        percent = (cached_games + games_processed) / total_games * 100
                        print(
                            f"Season {season} - Progress: {games_processed}/{remaining_games} games fetched | Game #{idx} | Overall: {percent:.1f}% complete"
                        )

                    break  # Success, exit retry loop

                except Exception as e:
                    error_msg = str(e).lower()

                    # Check if it's a rate limit error
                    if (
                        "429" in error_msg
                        or "rate limit" in error_msg
                        or "too many requests" in error_msg
                    ):
                        if attempt < max_retries - 1:
                            # Exponential backoff: 2s, 4s, 8s, etc.
                            backoff_delay = backoff_factor ** (attempt + 1)
                            print(
                                f"Rate limited on game {game_id}. Retrying in {backoff_delay}s... (attempt {attempt + 1}/{max_retries})"
                            )
                            time.sleep(backoff_delay)
                        else:
                            print(
                                f"Rate limit exceeded for game {game_id} after {max_retries} attempts. Skipping."
                            )
                    else:
                        print(f"Error processing game {game_id}: {e}")
                        break  # Non-rate-limit error, skip this game

        print(f"\n✓ Season {season} completed: {games_processed} new games fetched")
        print(f"Total injury records for season {season}: {len(season_injuries)}\n")

        all_seasons_injuries.append(season_injuries)

    # Combine all seasons
    if all_seasons_injuries:
        combined_injuries = pd.concat(all_seasons_injuries, ignore_index=True)
    else:
        combined_injuries = pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total injury records across all seasons: {len(combined_injuries)}")
    print(f"{'='*60}\n")

    return combined_injuries


def merge_injuries_with_games(game_log, injuries_log):
    """
    Merge injury data with game log by aggregating injuries to team-level.

    Args:
        game_log: DataFrame with game data (one row per team per game)
        injuries_log: DataFrame with injury data (one row per injured player)

    Returns:
        DataFrame with game_log plus INJURED_PLAYERS column
    """
    if injuries_log is None or injuries_log.empty:
        print("No injury data available, adding INJURED_PLAYERS column with 0s")
        result = game_log.copy()
        result['INJURED_PLAYERS'] = 0
        return result

    print(f"\n{'='*60}")
    print("MERGING INJURY DATA WITH GAME LOG")
    print(f"{'='*60}")

    # Aggregate injuries to team-level (count per team per game)
    injury_counts = injuries_log.groupby(['gameId', 'teamId']).size().reset_index(name='INJURED_PLAYERS')

    # Rename columns to match game_log format
    injury_counts = injury_counts.rename(columns={
        'gameId': 'GAME_ID',
        'teamId': 'TEAM_ID'
    })

    print(f"Unique games with injuries: {injury_counts['GAME_ID'].nunique()}")
    print(f"Total team-game injury records: {len(injury_counts)}")

    # Merge with game_log (left join to keep all games)
    result = game_log.merge(injury_counts, on=['GAME_ID', 'TEAM_ID'], how='left')

    # Fill NaN with 0 for games with no injuries
    result['INJURED_PLAYERS'] = result['INJURED_PLAYERS'].fillna(0).astype(int)

    games_with_injuries = (result['INJURED_PLAYERS'] > 0).sum()
    games_without_injuries = (result['INJURED_PLAYERS'] == 0).sum()

    print(f"Games with injuries: {games_with_injuries}")
    print(f"Games without injuries: {games_without_injuries}")
    print(f"Average injured players per team-game: {result['INJURED_PLAYERS'].mean():.2f}")
    print(f"{'='*60}\n")

    return result
