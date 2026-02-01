# NBA Data Analysis

## Which factors are most significant for NBA Wins and Losses

The code will:

- Fetch each season from NBA API (or use cached data)
- Combine all seasons into one dataset
- Add a `SEASON` column to identify which season each game belongs to
- Run XGBoost analysis on the combined dataset

## EDA Process Overview

The [eda.py](nba/eda.py) script does the following:

1. **Data Collection**
   - Fetches NBA game logs for configured seasons using `get_multiple_seasons()`
   - Uses caching system (local files) to avoid redundant API calls and check for new games
   - Each season is cached separately for faster subsequent runs

2. **Injury Data Collection**
   - Retrieves injury data for games using `get_season_game_injuries()`
   - Matches injury information with game data from boxscore endpoint

3. **Data Cleaning**
   - Cleans raw game log data via `clean_data()` with target column set to "WL" (Win/Loss)
   - Prepares data for analysis by handling missing values and formatting

4. **Feature Engineering**
   - Creates pre-game features using `create_pregame_features()`
   - Generates rolling averages and season statistics
   - Ensures only information available BEFORE each game is used (prevents data leakage)
   - Features are defined in features_config.json for easy configuration

5. **XGBoost Analysis**
   - Runs feature importance analysis using `find_top_features()`
   - Excludes PLUS_MINUS and MIN from features to find other contributing factors
   - Outputs top ~20 most important features by XGBoost gain
   - Reports train and test accuracy metrics

### XGBoost Feature Importance

- PLUS_MINUS (point differential) is excluded by default from the analysis as it's a (near?) perfect predictor of wins/losses, making it difficult to find other contributing factors.

### Considerations

- XGBoost automatically handles the larger dataset from multiple seasons
- Each season is cached separately for faster subsequent runs
- Data available contains post-game data, use only pre-game data for the Test Set
- Specific statistics are now defined in config file, for easier adding of explanatory variables

### Model Performance

###### 18DEC2025

- Rolling Window -- 10 days
- 57.0%

###### 01FEB2026

- Rolling Window -- 3 days
- 58.2%
