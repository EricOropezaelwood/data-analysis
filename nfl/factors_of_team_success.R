library(nflreadr)

# Set seasons for wanted Player stats
years <- c(2024, 2025)
# load player stats
team_stats <- load_team_stats()(seasons = years)
