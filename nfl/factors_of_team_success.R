library(nflreadr)
library(dplyr)

# Set seasons for analysis
years <- c(2022, 2023, 2024, 2025)

# Load team statistics data (weekly level)
team_stats_weekly <- load_team_stats(seasons = years)
# Load schedule data, which contains the results of the games (game level)
schedule_weekly <- load_schedules(seasons = years)

# Calculate wins per team, per game, from schedule data
# Schedule data has home_team and away_team;
#  result is from home team's perspective
# (positive = home win, negative = away win, 0 = tie)
schedule_played <- schedule_weekly %>%
  filter(!is.na(result))  # Remove games that haven't been played

# Bind any number of data frames by row, making a longer dataframe (from dplyr)
# Keep at game level: calculate win/loss/tie per team per game
team_wins_game <- bind_rows(
  # Home team results
  schedule_played %>%
    select(season, week, game_id, team = home_team, result) %>%
    mutate(
      win = if_else(result > 0, 1, 0),
      loss = if_else(result < 0, 1, 0),
      tie = if_else(result == 0, 1, 0)
    ),
  # Away team results
  schedule_played %>%
    select(season, week, game_id, team = away_team, result) %>%
    mutate(
      win = if_else(result < 0, 1, 0),
      loss = if_else(result > 0, 1, 0),
      tie = if_else(result == 0, 1, 0)
    )
)

# Combine team statistics (already at weekly/game level) with game results
# Join by season, week, and team to create game-level dataset for analysis
team_stats_game <- team_stats_weekly %>%
  left_join(team_wins_game, by = c("season", "week", "team"))

# Convert character columns to factors for modeling work
team_stats_game <- team_stats_game %>%
  mutate(across(where(is.character), as.factor))

# Fit multiple linear regression model
# Win as dependent variable (0 or 1 per game), all other numeric variables as predictors
lm_model <- lm(win ~ . - season - team - week - season_type - opponent_team - loss - tie - game_id - result,
               data = team_stats_game)
