library(nflreadr)
library(dplyr)

# Set seasons for analysis
years <- c(2022, 2023, 2024, 2025)

# Load team statistics data (weekly level)
team_stats_weekly <- load_team_stats(seasons = years)
# Load schedule data, which contains the results of the games (game level)
schedule_weekly <- load_schedules(seasons = years)

# Calculate wins per team, per season, from schedule data
# Schedule data has home_team and away_team;
#  result is from home team's perspective
# (positive = home win, negative = away win, 0 = tie)
schedule_played <- schedule_weekly %>%
  filter(!is.na(result))  # Remove games that haven't been played

# Bind any number of data frames by row, making a longer dataframe (from dplyr)
# Aggregate to season level: calculate total wins per team per season
team_wins_season <- bind_rows(
  # Home team results
  schedule_played %>%
    select(season, team = home_team, result) %>%
    mutate(
      wins = if_else(result > 0, 1, 0),
      losses = if_else(result < 0, 1, 0),
      ties = if_else(result == 0, 1, 0)
    ),
  # Away team results
  schedule_played %>%
    select(season, team = away_team, result) %>%
    mutate(
      wins = if_else(result < 0, 1, 0),
      losses = if_else(result > 0, 1, 0),
      ties = if_else(result == 0, 1, 0)
    )
) %>%
  group_by(season, team) %>%
  summarise(
    wins = sum(wins, na.rm = TRUE),
    losses = sum(losses, na.rm = TRUE),
    ties = sum(ties, na.rm = TRUE),
    .groups = "drop"
  )

# Aggregate team statistics from weekly to season level
# Sum weekly stats to get season totals (yards, points, turnovers, etc.)
team_stats_season_agg <- team_stats_weekly %>%
  group_by(season, team) %>%
  summarise(
    across(where(is.numeric), \(x) sum(x, na.rm = TRUE)),
    .groups = "drop"
  )

# Combine team statistics with wins data
# Join by season and team to create season-level dataset for analysis
team_stats_season <- team_stats_season_agg %>%
  left_join(team_wins_season, by = c("season", "team"))

# Convert character columns to factors for modeling work
team_stats_season <- team_stats_season %>%
  mutate(across(where(is.character), as.factor))

# Fit multiple linear regression model
# Wins as dependent variable, all other numeric variables as predictors
lm_model <- lm(wins ~ . - season - team - week - losses - ties,
               data = team_stats_season)
