"""
For the pre-game features.

Reads feature configuration from features_config.json and creates pre-game features
to avoid data leakage.

Note: This is because I saw the prediction rate was very high, which led to realizing the model
was using the actual game statistics to predict the outcome.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_feature_config(config_path='features_config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)


def create_pregame_features(data, config_path='features_config.json'):

    config = load_feature_config(config_path)

    print(f"\n{'='*60}")
    print("CREATING PRE-GAME FEATURES FROM CONFIG")
    print(f"{'='*60}")
    print(f"Config file: {config_path}")
    print(f"Rolling window: {config['rolling_window']} games")

    # Ensure GAME_DATE is datetime
    data = data.copy()
    data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'])

    # Sort by team and date (chronological order)
    data = data.sort_values(['TEAM_ID', 'GAME_DATE']).reset_index(drop=True)

    # Filter to only columns that exist in the data
    stat_cols = [col for col in config['stat_columns'] if col in data.columns]
    print(f"Computing pre-game features for {len(stat_cols)} statistics")

    # Create containers for new features
    pregame_features = {}
    rolling_window = config['rolling_window']
    season_col = config['season_column']

    # Calculate rolling averages and season averages for each stat
    for feature_type in config['feature_types']:
        if not feature_type['enabled']:
            continue

        if feature_type['name'] == 'rolling_avg':
            print(f"  Processing rolling averages...")
            for stat in stat_cols:
                rolling_avg = data.groupby('TEAM_ID')[stat].transform(
                    lambda x: x.shift(1).rolling(window=rolling_window, min_periods=1).mean()
                )
                pregame_features[f"{stat}{feature_type['suffix']}"] = rolling_avg
        else:
            raise ValueError(f"Invalid feature type: {feature_type['name']}")


    # Create DataFrame with all new features
    pregame_df = pd.DataFrame(pregame_features, index=data.index)

    # Combine with original data
    result = pd.concat([data, pregame_df], axis=1)

    # Drop the temporary WIN column
    if 'WIN' in result.columns:
        result = result.drop(columns=['WIN'])

    print(f"\nAdded {len(pregame_features)} pre-game features")
    print(f"{'='*60}\n")

    return result


def get_feature_columns(data, config_path='features_config.json'):
    """
    Get list of pre-game feature columns from config.

    Note: Always returns pre-game features only to avoid data leakage.
    """
    config = load_feature_config(config_path)

    # Build list of pre-game feature columns from config
    feature_cols = []

    for feature_type in config['feature_types']:
        if not feature_type['enabled']:
            continue

        if feature_type['name'] in ['rolling_avg', 'season_avg']:
            # Add stat columns with the appropriate suffix
            suffix = feature_type['suffix']
            for stat in config['stat_columns']:
                col_name = f"{stat}{suffix}"
                if col_name in data.columns:
                    feature_cols.append(col_name)
        else:
            # Add special columns like WIN_PCT_SEASON, GAMES_PLAYED_SEASON
            col_name = feature_type.get('column_name')
            if col_name and col_name in data.columns:
                feature_cols.append(col_name)

    print(f"\nUsing {len(feature_cols)} pre-game features from config (no data leakage)")
    return feature_cols


def compare_pregame_vs_postgame(data, game_date, team_name):

    game = data[(data['GAME_DATE'] == game_date) &
                ((data['TEAM_ABBREVIATION'] == team_name) |
                 (data['TEAM_NAME'] == team_name))].iloc[0]

    print(f"\n{'='*60}")
    print(f"GAME: {team_name} on {game_date}")
    print(f"OUTCOME: {game['WL']}")
    print(f"{'='*60}\n")

    print("PRE-GAME FEATURES (what model sees):")
    print(f"  PTS_SEASON_AVG: {game.get('PTS_SEASON_AVG', 'N/A'):.1f}")
    print(f"  FG_PCT_SEASON_AVG: {game.get('FG_PCT_SEASON_AVG', 'N/A'):.3f}")
    print(f"  AST_SEASON_AVG: {game.get('AST_SEASON_AVG', 'N/A'):.1f}")
    print(f"  REB_SEASON_AVG: {game.get('REB_SEASON_AVG', 'N/A'):.1f}")
    print(f"  WIN_PCT_SEASON: {game.get('WIN_PCT_SEASON', 'N/A'):.3f}")

    print("\nACTUAL GAME STATS (hidden from model):")
    print(f"  PTS: {game.get('PTS', 'N/A')}")
    print(f"  FG_PCT: {game.get('FG_PCT', 'N/A'):.3f}")
    print(f"  AST: {game.get('AST', 'N/A')}")
    print(f"  REB: {game.get('REB', 'N/A')}")

    print(f"\n{'='*60}\n")
