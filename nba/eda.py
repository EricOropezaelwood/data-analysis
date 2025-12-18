from nba_api.stats.endpoints import leaguegamelog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import clean_data
from preprocessing import create_pregame_features, compare_pregame_vs_postgame
from xgboost_analysis import find_top_features
from save_results import save_test_results_to_csv
import pickle
from pathlib import Path
from datetime import datetime
import time


# Set modern seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 8)

# from nba_api
def get_league_game_log(season, use_cache=True, force_refresh=False):
    cache_file = Path(f'game_log_{season}.pkl')
    cached_data = None

    # Load existing cache if available (rate limiting...)
    if use_cache and cache_file.exists() and not force_refresh:
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        # check if game(s) are cached
        if 'GAME_DATE' in cached_data.columns:
            latest_date = pd.to_datetime(cached_data['GAME_DATE']).max()
            total_cached_games = len(cached_data)
            print(f"  Cached: {total_cached_games} games (latest: {latest_date.strftime('%Y-%m-%d')})")

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
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Referer': 'https://www.stats.nba.com/'
            }
        )
        new_data = game_log.get_data_frames()[0]

        # If we have cached data, merge it with new data
        if cached_data is not None and 'GAME_DATE' in new_data.columns:
            # Convert dates for comparison
            new_data['GAME_DATE'] = pd.to_datetime(new_data['GAME_DATE'])
            cached_data['GAME_DATE'] = pd.to_datetime(cached_data['GAME_DATE'])

            # Find games not in cache (by GAME_ID if available, otherwise by date)
            if 'GAME_ID' in new_data.columns:
                existing_game_ids = set(cached_data['GAME_ID'])
                new_games = new_data[~new_data['GAME_ID'].isin(existing_game_ids)]
            else:
                latest_cached_date = cached_data['GAME_DATE'].max()
                new_games = new_data[new_data['GAME_DATE'] > latest_cached_date]

            if len(new_games) > 0:
                print(f"  Found {len(new_games)} new games since cache")
                df = pd.concat([cached_data, new_games], ignore_index=True)
                df = df.sort_values('GAME_DATE').reset_index(drop=True)
            else:
                print(f"  No new games found, using cached data")
                df = cached_data
        else:
            df = new_data

        # Save updated cache
        with open(cache_file, 'wb') as f:
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
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise


def get_multiple_seasons(seasons, use_cache=True, force_refresh=False):

    if isinstance(seasons, (str, int)):
        seasons = [seasons]

    print(f"\nFetching data for {len(seasons)} season(s): {', '.join(map(str, seasons))}")
    print("=" * 60)

    all_data = []

    for season in seasons:
        try:
            df = get_league_game_log(season, use_cache=use_cache, force_refresh=force_refresh)
            df['SEASON'] = str(season)
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


# normalize the data
#  standardization (Z-score normalization)
# https://www.geeksforgeeks.org/data-analysis/z-score-normalization-definition-and-examples/
def normalize_data(data):
    # Create a copy to avoid modifying the original data
    normalized = data.copy()
    
    # Select only numeric columns
    numeric_cols = normalized.select_dtypes(include=[np.number]).columns
    
    # Apply standardization: (x - mean) / std
    normalized[numeric_cols] = normalized[numeric_cols].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    
    return normalized

# Visually see the correlations between the Response variable and the Explanatory variables
def plot_correlations(correlations, target_col='WL', save_path=None):
    # Create figure with modern styling
    fig, ax = plt.subplots(figsize=(10, max(8, len(correlations) * 0.4)))
    
    # Create color map: positive correlations in blue, negative in red
    colors = ['#4393c3' if x > 0 else '#d6604d' for x in correlations.values]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.8)
    
    # Customize plot
    ax.set_yticks(range(len(correlations)))
    ax.set_yticklabels(correlations.index, fontsize=10)
    ax.set_xlabel('Correlation with ' + target_col, fontsize=12, fontweight='bold')
    ax.set_title(f'Correlation of Variables with {target_col}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(correlations.items()):
        ax.text(val, i, f' {val:.3f}', 
                va='center', fontsize=9,
                color='white' if abs(val) > 0.3 else 'black',
                fontweight='bold' if abs(val) > 0.3 else 'normal')
    
    # Set x-axis limits to show full range
    x_margin = max(abs(correlations.min()), abs(correlations.max())) * 0.1
    ax.set_xlim(correlations.min() - x_margin, correlations.max() + x_margin)
    
    # Invert y-axis so highest correlation is at top
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4393c3', alpha=0.8, label='Positive correlation'),
        Patch(facecolor='#d6604d', alpha=0.8, label='Negative correlation')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCorrelation plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def find_significant_variables(data, target_col='WL', top_n=20, plot=True, save_path=None):
    # Create a copy to avoid modifying original data
    data_copy = data.copy()
    
    # Convert target column to numeric if it's string (e.g., 'W'/'L' -> 1/0)
    if data_copy[target_col].dtype == 'object':
        # Map W to 1, L to 0
        data_copy[target_col] = data_copy[target_col].map({'W': 1, 'L': 0})
    
    # Select only numeric columns
    numeric_cols = data_copy.select_dtypes(include=[np.number]).columns
    
    # Calculate correlations with target column
    correlations = data_copy[numeric_cols].corrwith(data_copy[target_col])
    
    # Sort by absolute correlation (descending)
    correlations = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    )
    
    # Remove the target column itself if present
    correlations = correlations.drop(labels=[target_col], errors='ignore')
    # Remove obvious columns that shouldn't be used as features
    correlations = correlations.drop(labels=['TEAM_ID', 'SEASON'], errors='ignore')
    
    # Get top N correlations
    top_correlations = correlations.head(top_n)
    
    # Plot correlations if requested
    if plot:
        plot_correlations(top_correlations, target_col=target_col, save_path=save_path)
    
    return top_correlations


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

    # Clean the data
    cleaned_data = clean_data(game_log, target_col='WL')

    # Create pre-game features (rolling averages, season stats, etc.)
    # This ensures we only use information available BEFORE each game
    # Features are defined in features_config.json
    cleaned_data = create_pregame_features(cleaned_data)

    # Show date range in cleaned data
    if 'GAME_DATE' in cleaned_data.columns:
        dates = pd.to_datetime(cleaned_data['GAME_DATE'])
        print(f"\n{'='*60}")
        print("CLEANED DATA DATE RANGE")
        print(f"{'='*60}")
        print(f"Total games: {len(cleaned_data)}")
        print(f"Earliest game: {dates.min().strftime('%Y-%m-%d')}")
        print(f"Latest game: {dates.max().strftime('%Y-%m-%d')}")
        print(f"Date span: {(dates.max() - dates.min()).days} days")

        # Show games by season
        if 'SEASON' in cleaned_data.columns:
            print(f"\nGames by season:")
            for season in sorted(cleaned_data['SEASON'].unique()):
                season_data = cleaned_data[cleaned_data['SEASON'] == season]
                season_dates = pd.to_datetime(season_data['GAME_DATE'])
                print(f"  {season}: {len(season_data)} games ({season_dates.min().strftime('%Y-%m-%d')} to {season_dates.max().strftime('%Y-%m-%d')})")
        print(f"{'='*60}\n")
    
    # Find significant variables (before normalization)
    # print("Finding Significant Variables for WL")
    # significant_vars = find_significant_variables(
    #     cleaned_data,
    #     target_col='WL',
    #     top_n=20,
    #     plot=True,
    #     save_path='correlations_with_wl.png'
    # )
    # print("\nTop 20 variables most correlated with WL:")
    # print(significant_vars)

    # XGBoost analysis for feature importance
    print("\n" + "="*60)
    print("XGBoost Feature Importance Analysis")
    print("="*60)

    # print("\n1. ALL FEATURES:")
    # top_features, train_acc, test_acc = find_top_features(
    #     cleaned_data,
    #     target_col='WL',
    #     top_n=20
    # )
    # print(f"Model Accuracy - Train: {train_acc:.3f}, Test: {test_acc:.3f}")
    # print("\nTop features by XGBoost gain:")
    # for feature, gain in top_features.items():
    #     print(f"  {feature}: {gain:.2f}")

    print("\n2. USING PRE-GAME FEATURES (no data leakage):")
    top_features_no_pm, train_acc_no_pm, test_acc_no_pm, X_test, y_test, y_pred, y_pred_proba = find_top_features(
        cleaned_data,
        target_col='WL',
        top_n=20,
        exclude_features=['PLUS_MINUS', 'MIN']
    )

    print(f"Model Accuracy - Train: {train_acc_no_pm:.3f}, Test: {test_acc_no_pm:.3f}")
    print("\nTop features by XGBoost gain:")
    for feature, gain in top_features_no_pm.items():
        print(f"  {feature}: {gain:.2f}")

    # Save test results to CSV (for me to explore)
    csv_file = save_test_results_to_csv(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        train_acc=train_acc_no_pm,
        test_acc=test_acc_no_pm,
        original_data=cleaned_data,
        output_dir='test_results'
    )

    # Normalize the data

