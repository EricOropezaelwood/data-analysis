library(nflreadr)

# Set seasons for wanted Player stats
years <- c(2024, 2025)
# load player stats
player_stats = load_player_stats(seasons=years)

