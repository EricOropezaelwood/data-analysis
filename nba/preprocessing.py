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


def load_feature_config(config_path="features_config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def create_pregame_features(data, config_path="features_config.json"):

    config = load_feature_config(config_path)

    print(f"\n{'='*60}")
    print("CREATING PRE-GAME FEATURES FROM CONFIG")
    print(f"{'='*60}")
    print(f"Config file: {config_path}")
    print(f"Rolling window: {config['rolling_window']} games")

    # Ensure GAME_DATE is datetime
    data = data.copy()
    data["GAME_DATE"] = pd.to_datetime(data["GAME_DATE"])

    # Sort by team and date (chronological order)
    data = data.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    # Filter to only columns that exist in the data
    stat_cols = [col for col in config["stat_columns"] if col in data.columns]
    print(f"Computing pre-game features for {len(stat_cols)} statistics")

    # Create containers for new features
    pregame_features = {}
    rolling_window = config["rolling_window"]
    season_col = config["season_column"]

    # Calculate rolling averages and season averages for each stat
    for feature_type in config["feature_types"]:
        if not feature_type["enabled"]:
            continue

        if feature_type["name"] == "rolling_avg":
            print(f"  Processing rolling averages...")
            for stat in stat_cols:
                rolling_avg = data.groupby("TEAM_ID")[stat].transform(
                    lambda x: x.shift(1)
                    .rolling(window=rolling_window, min_periods=1)
                    .mean()
                )
                pregame_features[f"{stat}{feature_type['suffix']}"] = rolling_avg
        else:
            raise ValueError(f"Invalid feature type: {feature_type['name']}")

    # Create DataFrame with all new features
    pregame_df = pd.DataFrame(pregame_features, index=data.index)

    # Combine with original data
    result = pd.concat([data, pregame_df], axis=1)

    # Drop the temporary WIN column
    if "WIN" in result.columns:
        result = result.drop(columns=["WIN"])

    print(f"\nAdded {len(pregame_features)} pre-game features")
    print(f"{'='*60}\n")

    return result


def add_opponent_features(data, config_path="features_config.json"):
    config = load_feature_config(config_path)

    print("========================")
    print("ADDING OPPONENT FEATURES")
    print("========================")

    # create copy of input data
    data = data.copy()

    # build list of columns to copy from opponent
    opp_columns = []

    # add rolling average columns
    for feature_type in config["feature_types"]:
        if not feature_type["enabled"]:
            continue
        if feature_type["name"] == "rolling_avg":
            suffix = feature_type["suffix"]
            for stat in config["stat_columns"]:
                col_name = f"{stat}{suffix}"
                if col_name in data.columns:
                    opp_columns.append(col_name)

    # add additional features (like INJURED_PLAYERS)
    for col_name in config.get("additional_features", []):
        if col_name in data.columns:
            opp_columns.append(col_name)

    print(f"Copying {len(opp_columns)} features from opponent")

    # create a lookup dataframe with GAME_ID, TEAM_ID, and the columns we want
    lookup_cols = ["GAME_ID", "TEAM_ID"] + opp_columns
    opponent_data = data[lookup_cols].copy()

    # rename columns with OPP_ prefix (except GAME_ID and TEAM_ID)
    rename_map = {col: f"OPP_{col}" for col in opp_columns}
    opponent_data = opponent_data.rename(columns=rename_map)

    # for each game, we need to match each team with their opponent
    # self-merge on GAME_ID where TEAM_ID differs
    result = data.merge(opponent_data, on="GAME_ID", suffixes=("", "_opp"))

    # keep only rows where the opponent is different (removes self-matches)
    result = result[result["TEAM_ID"] != result["TEAM_ID_opp"]]

    # drop the opponent's TEAM_ID column
    result = result.drop(columns=["TEAM_ID_opp"])

    # reset index
    result = result.reset_index(drop=True)

    opp_feature_names = list(rename_map.values())
    print(f"Added {len(opp_feature_names)} opponent features")
    print(f"Example features: {', '.join(opp_feature_names[:3])}...")
    print(f"{'='*60}\n")

    return result


def get_feature_columns(data, config_path="features_config.json"):
    """
    Get list of pre-game feature columns from config.

    Note: Always returns pre-game features only to avoid data leakage.
    """
    config = load_feature_config(config_path)

    # Build list of pre-game feature columns from config
    feature_cols = []

    for feature_type in config["feature_types"]:
        if not feature_type["enabled"]:
            continue

        if feature_type["name"] in ["rolling_avg", "season_avg"]:
            # Add stat columns with the appropriate suffix
            suffix = feature_type["suffix"]
            for stat in config["stat_columns"]:
                col_name = f"{stat}{suffix}"
                if col_name in data.columns:
                    feature_cols.append(col_name)
        else:
            # Add special columns like WIN_PCT_SEASON, GAMES_PLAYED_SEASON
            col_name = feature_type.get("column_name")
            if col_name and col_name in data.columns:
                feature_cols.append(col_name)

    # Add additional features (direct columns like INJURED_PLAYERS)
    for col_name in config.get("additional_features", []):
        if col_name in data.columns:
            feature_cols.append(col_name)

    # add opponent features (OPP_ prefixed versions of all features above)
    opp_features = [
        f"OPP_{col}" for col in feature_cols if f"OPP_{col}" in data.columns
    ]
    feature_cols.extend(opp_features)

    print(
        f"\nUsing {len(feature_cols)} pre-game features from config (no data leakage)"
    )
    return feature_cols


def compare_pregame_vs_postgame(data, game_date, team_name):

    game = data[
        (data["GAME_DATE"] == game_date)
        & ((data["TEAM_ABBREVIATION"] == team_name) | (data["TEAM_NAME"] == team_name))
    ].iloc[0]

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
