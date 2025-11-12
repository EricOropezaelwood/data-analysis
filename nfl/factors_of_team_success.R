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

# Data Cleaning for Zeros and NaNs
# ============================================================

cat("\n=== Data Cleaning Report ===\n")
cat("Initial dimensions:", nrow(team_stats_game), "rows,", ncol(team_stats_game), "columns\n\n")

# Remove rows with missing outcome variable
team_stats_clean <- team_stats_game %>%
  filter(!is.na(win))

# Identify and handle problematic columns
numeric_cols <- team_stats_clean %>%
  select(where(is.numeric)) %>%
  select(-win, -loss, -tie)

# THIS REALLY HELPS THE NAN ISSUES, JUST SAYIN' <<<<------------------------------------
# Check for columns with high proportion of NAs (>50%)
high_na_cols <- team_stats_clean %>%
  summarise(across(everything(), ~mean(is.na(.)))) %>%
  select(where(~. > 0.5)) %>%
  names()

if(length(high_na_cols) > 0) {
  cat("Removing", length(high_na_cols), "high-NA columns (>50% missing):",
      paste(head(high_na_cols, 10), collapse = ", "))
  if(length(high_na_cols) > 10) cat(" ...")
  cat("\n")
  team_stats_clean <- team_stats_clean %>%
    select(-all_of(high_na_cols))
}

# Handle columns where all non-NA values are zero
# These columns have no predictive power but don't have variance (all zeros)
all_zero_cols <- team_stats_clean %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), ~all(. == 0 | is.na(.)))) %>%
  select(where(~.)) %>%
  names()

if(length(all_zero_cols) > 0) {
  cat("Removing", length(all_zero_cols), "all-zero columns:",
      paste(head(all_zero_cols, 10), collapse = ", "))
  if(length(all_zero_cols) > 10) cat(" ...")
  cat("\n")
  team_stats_clean <- team_stats_clean %>%
    select(-all_of(all_zero_cols))
}

# Handle columns with mostly zeros (>95% zeros)
# Option: Flag for potential removal or transformation
mostly_zero_cols <- team_stats_clean %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), ~mean(. == 0, na.rm = TRUE))) %>%
  select(where(~. > 0.95)) %>%
  names()

if(length(mostly_zero_cols) > 0) {
  cat("Warning:", length(mostly_zero_cols), "columns are >95% zeros (keeping but may cause issues):",
      paste(head(mostly_zero_cols, 5), collapse = ", "))
  if(length(mostly_zero_cols) > 5) cat(" ...")
  cat("\n")
  # Optionally remove these: team_stats_clean <- team_stats_clean %>% select(-all_of(mostly_zero_cols))
}

# Remove rows with any remaining NAs in predictors
rows_before <- nrow(team_stats_clean)
team_stats_clean <- team_stats_clean %>%
  na.omit()
rows_removed <- rows_before - nrow(team_stats_clean)

if(rows_removed > 0) {
  cat("Removed", rows_removed, "rows with NA values\n")
}

cat("\nFinal dimensions:", nrow(team_stats_clean), "rows,", ncol(team_stats_clean), "columns\n")
cat("=== End Cleaning Report ===\n\n")

# Fit Linear Regression Model
# ---------------------------

# Win as dependent variable (0 or 1 per game), all other numeric variables as predictors
lm_model <- lm(win ~ . - season - team - week - season_type - opponent_team - loss - tie - game_id - result,
               data = team_stats_clean)
