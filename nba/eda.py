from nba_api.stats.endpoints import leaguegamelog
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning import clean_data

# Set modern seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 8)

# from nba_api 
def get_league_game_log(season):
    game_log = leaguegamelog.LeagueGameLog(season=season)
    return game_log.get_data_frames()[0]

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
    # Remove obvious columns
    correlations = correlations.drop(labels=['TEAM_ID'])
    
    # Get top N correlations
    top_correlations = correlations.head(top_n)
    
    # Plot correlations if requested
    if plot:
        plot_correlations(top_correlations, target_col=target_col, save_path=save_path)
    
    return top_correlations


if __name__ == "__main__":
    # get the game log for the 2025 season
    game_log = get_league_game_log(2025)
    
    # Clean the data
    cleaned_data = clean_data(game_log, target_col='WL')
    
    # Find significant variables (before normalization)
    print("Finding Significant Variables for WL")
    significant_vars = find_significant_variables(
        cleaned_data, 
        target_col='WL', 
        top_n=20, 
        plot=True,
        save_path='correlations_with_wl.png'
    )
    print("\nTop 20 variables most correlated with WL:")
    print(significant_vars)
    
    # Normalize the data
 
