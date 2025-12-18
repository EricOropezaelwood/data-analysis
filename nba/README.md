# NBA Data Analysis

## Which factors are most significant for NBA Wins and Losses

The code will:
- Fetch each season from NBA API (or use cached data)
- Combine all seasons into one dataset
- Add a `SEASON` column to identify which season each game belongs to
- Run XGBoost analysis on the combined dataset

### XGBoost Feature Importance
- PLUS_MINUS (point differential) is excluded by default from the analysis as it's a (near?) perfect predictor of wins/losses, making it difficult to find other contributing factors.

### Considerations
- XGBoost automatically handles the larger dataset from multiple seasons
- Each season is cached separately for faster subsequent runs
- Data available contains post-game data, use only pre-game data for the Test Set
- Specific statistics are now defined in config file, for easier adding of explanatory variables

### Model Performance

###### 18DEC2025
- 53.6%