library(nflreadr)
library(dplyr)
library(ggplot2)
library(corrplot)
# Additional libraries for logistic regression evaluation (following Statology guide)
library(pscl)  # For McFadden's R²
library(caret)  # For variable importance

# Disable scientific notation for model summary
options(scipen = 999)


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
  filter(!is.na(result)) # Remove games that haven't been played

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
cat(
  "Initial dimensions:",
  nrow(team_stats_game), "rows,",
  ncol(team_stats_game), "columns\n\n"
)

# Remove rows with missing outcome variable
team_stats_clean <- team_stats_game %>%
  filter(!is.na(win))

# Identify and handle problematic columns
numeric_cols <- team_stats_clean %>%
  select(where(is.numeric)) %>%
  select(-win, -loss, -tie)

# THIS REALLY HELPS THE NAN ISSUES, JUST SAYIN' <<<<--------------
# Check for columns with high proportion of NAs (>50%)
high_na_cols <- team_stats_clean %>%
  summarise(across(everything(), ~ mean(is.na(.)))) %>%
  select(where(~ . > 0.5)) %>%
  names()

if (length(high_na_cols) > 0) {
  cat(
    "Removing", length(high_na_cols), "high-NA columns (>50% missing):",
    paste(head(high_na_cols, 10), collapse = ", ")
  )
  if (length(high_na_cols) > 10) cat(" ...")
  cat("\n")
  team_stats_clean <- team_stats_clean %>%
    select(-all_of(high_na_cols))
}

# Handle columns where all non-NA values are zero
# These columns have no predictive power but don't have variance (all zeros)
all_zero_cols <- team_stats_clean %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), ~ all(. == 0 | is.na(.)))) %>%
  select(where(~.)) %>%
  names()

if (length(all_zero_cols) > 0) {
  cat(
    "Removing", length(all_zero_cols), "all-zero columns:",
    paste(head(all_zero_cols, 10), collapse = ", ")
  )
  if (length(all_zero_cols) > 10) cat(" ...")
  cat("\n")
  team_stats_clean <- team_stats_clean %>%
    select(-all_of(all_zero_cols))
}

# Handle columns with mostly zeros (>95% zeros)
# Option: Flag for potential removal or transformation
mostly_zero_cols <- team_stats_clean %>%
  select(where(is.numeric)) %>%
  summarise(across(everything(), ~ mean(. == 0, na.rm = TRUE))) %>%
  select(where(~ . > 0.95)) %>%
  names()

if (length(mostly_zero_cols) > 0) {
  cat(
    "Warning:", length(mostly_zero_cols),
    "columns are >95% zeros (keeping but may cause issues):\n\n",
    paste(head(mostly_zero_cols, 5), collapse = ", ")
  )
  if (length(mostly_zero_cols) > 5) cat(" ...")
  cat("\n")
  # Optionally remove these: team_stats_clean <- team_stats_clean
  # %>% select(-all_of(mostly_zero_cols))
}

# Remove rows with any remaining NAs in predictors
rows_before <- nrow(team_stats_clean)
team_stats_clean <- team_stats_clean %>%
  na.omit()
rows_removed <- rows_before - nrow(team_stats_clean)

if (rows_removed > 0) {
  cat("Removed", rows_removed, "rows with NA values\n")
}

cat(
  "\nFinal dimensions:",
  nrow(team_stats_clean), "rows,",
  ncol(team_stats_clean), "columns\n"
)
cat("=== End Cleaning Report ===\n\n")

# Correlation Heatmap
# ============================================================

# Define variables to strictly exclude from predictors
# Explicitly remove 'win' and other proxies to ensure no circular logic.
excluded_vars <- c(
  "season", "team", "week", "season_type", "opponent_team",
  "loss", "tie", "game_id", "result", "win",
  "gwfg_made", "gwfg_att", "gwfg_missed", "gwfg_blocked", "gwfg_distance",
  "pat_made", "pat_att"
)

# Select only numeric columns for correlation analysis
numeric_data <- team_stats_clean %>%
  select(where(is.numeric)) %>%
  select(-any_of(excluded_vars))

# Calculate correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")

# Create correlation heatmap focusing on correlations with 'win'
# "win" is the response variable, so we'll use correlations with "win" to select
# explanatory variables.
# NOTE: We need to calculate correlation with 'win' separately since we removed it from numeric_data
win_correlations <- cor(
  team_stats_clean %>%
    select(where(is.numeric)),
  team_stats_clean$win,
  use = "complete.obs"
)
win_correlations <- win_correlations[, 1] # Convert to vector
# Remove excluded vars from win_correlations
win_correlations <- win_correlations[!names(win_correlations) %in% excluded_vars]
win_correlations <- win_correlations[order(abs(win_correlations), decreasing = TRUE)]

# Select top explanatory variables based on correlation with "win"
# Top 30 explanatory variables
top_explanatory_vars <- names(win_correlations)[seq_len(min(30, length(win_correlations)))]

# Subset correlation matrix to explanatory variables only
cor_subset <- cor_matrix[top_explanatory_vars, top_explanatory_vars]

# Verify "win" is NOT in the subset (it's the response variable)
cat("\n=== Correlation Heatmap Info ===\n")
cat("Total explanatory variables in heatmap:", nrow(cor_subset), "\n")
cat(
  "'win' excluded from heatmap (response variable):",
  !"win" %in% rownames(cor_subset), "\n"
)
cat("\nTop 10 explanatory variables most correlated with 'win':\n")
print(head(win_correlations[names(win_correlations) != "win"], 10))
cat("\nCorrelations with 'win' (response variable):\n")
print(win_correlations[top_explanatory_vars])
cat("================================\n\n")

# Create heatmap using base R heatmap() function with clustering
# This automatically groups similar variables together using hierarchical
# clustering and displays dendrograms to show relationships
png("/Users/quixote/Coding/data-analysis/nfl/correlation_heatmap.png",
  width = 1400, height = 1300, res = 150
)

# Custom color palette (red for negative, white for zero, blue for positive)
heatmap_colors <- colorRampPalette(
  c("#67001f", "#d6604d", "#f7f7f7", "#4393c3", "#053061")
)(200)

# Set margins to ensure title is visible
par(oma = c(0, 0, 3, 0)) # Outer margins: bottom, left, top, right

# Create heatmap with clustering
# Rowv and Colv = TRUE enables hierarchical clustering on both rows and columns
# scale = "none" because correlation values are already normalized (-1 to 1)
# margins adjusted for better label visibility
heatmap(cor_subset,
  Rowv = TRUE, # Cluster rows (variables)
  Colv = TRUE, # Cluster columns (variables)
  scale = "none", # Don't scale (correlations already normalized)
  col = heatmap_colors, # Color scheme
  symm = TRUE, # Symmetric matrix (correlation is symmetric)
  margins = c(15, 12), # Margins: bottom (x-axis), left (y-axis)
  cexRow = 0.7, # Row label size
  cexCol = 0.8, # Column label size
  main = paste0(
    "Correlation Heatmap: Explanatory Variables ",
    "(Top 30 Correlated with Win)\nwith Hierarchical Clustering"
  )
)

dev.off()

cat("Correlation heatmap saved to: nfl/correlation_heatmap.png\n")
cat("(Shows correlations BETWEEN explanatory variables)\n\n")

# Create a visualization showing correlations WITH "win" (the response variable)
# This shows how strongly each explanatory variable correlates with "win"
png("/Users/quixote/Coding/data-analysis/nfl/correlations_with_win.png",
  width = 1200, height = 1000, res = 150
)

# Get correlations with "win" for the top explanatory variables
win_cor_subset <- win_correlations[top_explanatory_vars]
win_cor_subset <- win_cor_subset[order(abs(win_cor_subset), decreasing = TRUE)]

# Create bar plot showing correlations with "win"
par(mar = c(8, 10, 4, 4)) # Margins: bottom, left, top, right
barplot(win_cor_subset,
  horiz = TRUE,
  las = 1, # Horizontal labels
  # Blue for positive, red for negative
  col = ifelse(win_cor_subset > 0, "#4393c3", "#d6604d"),
  xlab = "Correlation with Win",
  main = "Correlation of Explanatory Variables with Win (Response Variable)",
  cex.names = 0.7,
  xlim = c(-1, 1)
)
abline(v = 0, lty = 2, col = "gray50") # Reference line at zero

dev.off()

cat("Correlations with 'win' vis saved to: nfl/correlations_with_win.png\n")
cat("(Shows how strongly each explanatory variable correlates with 'win')\n\n")

# Prepare dataset for logistic regression using top explanatory variables
# Select only the top explanatory variables and the outcome variable
model_data <- team_stats_clean %>%
  select(all_of(top_explanatory_vars), win)

cat("\n=== Logistic Regression Model Setup ===\n")
cat("Number of observations:", nrow(model_data), "\n")
cat("Number of explanatory variables:", length(top_explanatory_vars), "\n")
cat("Top explanatory variables:\n")
print(top_explanatory_vars)
cat("\n")

# Step 2: Create Training and Test Samples
# Use 70% of dataset as training set and remaining 30% as testing set
set.seed(103) # Set seed for reproducibility
sample <- sample(c(TRUE, FALSE),
                 nrow(model_data),
                 replace = TRUE,
                 prob = c(0.7, 0.3))
train_data <- model_data[sample, ]
test_data <- model_data[!sample, ]

# Step 3: Fit the Logistic Regression Model
cat("=== Step 3: Fitting Logistic Regression Model ===\n")
cat("Model formula: win ~ [top explanatory variables]\n")
cat("Target: Predict whether a team wins (1) or loses/ties (0) based on team statistics\n\n")

# Fit logistic regression model
log_model <- glm(win ~ ., data = train_data, family = "binomial")



# View model summary
cat("=== Model Summary ===\n")
summary(log_model)

# Step 4: Assess Model Fit
cat("\n=== Step 4: Assessing Model Fit ===\n")

# McFadden's R² (values over 0.40 indicate good fit)
cat("\n--- McFadden's R² ---\n")
mcfadden_r2 <- pscl::pR2(log_model)["McFadden"]
cat("McFadden's R²:", round(mcfadden_r2, 4), "\n")
if (mcfadden_r2 > 0.40) {
  cat("Interpretation: Model fits the data very well (R² > 0.40)\n")
} else if (mcfadden_r2 > 0.20) {
  cat("Interpretation: Model has moderate fit (R² between 0.20-0.40)\n")
} else {
  cat("Interpretation: Model has limited predictive power (R² < 0.20)\n")
}

# Variable Importance
cat("\n--- Variable Importance ---\n")
# compute the importance of each predictor variable in the model
var_importance <- caret::varImp(log_model)
cat("Variable importance (higher values = more important):\n")
print(var_importance)

# Step 5: Use the Model to Make Predictions
cat("\n=== Step 5: Making Predictions ===\n")

# Predictions on test set
test_pred_probs <- predict(log_model, newdata = test_data, type = "response")
test_pred_class <- ifelse(test_pred_probs > 0.5, 1, 0)

# Create confusion matrix
cat("\n--- Confusion Matrix (Test Set) ---\n")
confusion_matrix <- table(Actual = test_data$win, Predicted = test_pred_class)
print(confusion_matrix)

# Calculate accuracy
test_accuracy <- mean(test_pred_class == test_data$win, na.rm = TRUE)
cat("\nTest Accuracy:", round(test_accuracy * 100, 2), "%\n")
