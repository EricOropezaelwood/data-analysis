import pandas as pd
from pathlib import Path
from datetime import datetime
from rich import print
from rich.table import Table
from rich.console import Console


def save_test_results_to_csv(X_test, y_test, y_pred, y_pred_proba, train_acc, test_acc,
                              original_data=None, output_dir='test_results'):

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # using only the date, not the time
    timestamp = datetime.now().strftime('%Y%m%d')
    filename = f"test_results_{timestamp}.csv"
    filepath = output_path / filename

    # Create results dataframe starting with key columns
    results = pd.DataFrame()

    # Add team name first if available
    if original_data is not None:
        test_indices = X_test.index
        if 'TEAM_ABBREVIATION' in original_data.columns:
            results['TEAM_NAME'] = original_data.loc[test_indices, 'TEAM_ABBREVIATION'].values
        elif 'TEAM_NAME' in original_data.columns:
            results['TEAM_NAME'] = original_data.loc[test_indices, 'TEAM_NAME'].values
        else:
            results['TEAM_NAME'] = 'Unknown'
    else:
        results['TEAM_NAME'] = 'Unknown'

    # Add game date if available
    if original_data is not None:
        test_indices = X_test.index
        if 'GAME_DATE' in original_data.columns:
            results['GAME_DATE'] = original_data.loc[test_indices, 'GAME_DATE'].values

    # Add prediction and outcome columns
    results['PREDICTED_OUTCOME'] = ['W' if p == 1 else 'L' for p in y_pred]
    results['ACTUAL_OUTCOME'] = ['W' if y == 1 else 'L' for y in y_test]
    results['CORRECT_PREDICTION'] = (y_test.values == y_pred)
    results['WIN_PROBABILITY'] = y_pred_proba[:, 1]
    results['LOSS_PROBABILITY'] = y_pred_proba[:, 0]

    # Add game metadata if available
    if original_data is not None:
        test_indices = X_test.index
        if 'GAME_ID' in original_data.columns:
            results['GAME_ID'] = original_data.loc[test_indices, 'GAME_ID'].values
        if 'TEAM_ID' in original_data.columns:
            results['TEAM_ID'] = original_data.loc[test_indices, 'TEAM_ID'].values
        if 'SEASON' in original_data.columns:
            results['SEASON'] = original_data.loc[test_indices, 'SEASON'].values

    # Add all feature columns
    for col in X_test.columns:
        results[col] = X_test[col].values

    # Add metadata at the end
    results['MODEL_TRAIN_ACC'] = train_acc
    results['MODEL_TEST_ACC'] = test_acc
    results['TEST_TIMESTAMP'] = timestamp

    # Save to CSV
    results.to_csv(filepath, index=False)

    # Calculate and display accuracy stats
    accuracy = results['CORRECT_PREDICTION'].mean()
    total_games = len(results)
    correct_predictions = results['CORRECT_PREDICTION'].sum()

    # Create rich table with all results
    console = Console()
    table = Table(title="[bold cyan]Test Results Saved to CSV[/bold cyan]",
                  title_style="bold cyan",
                  show_header=True,
                  header_style="bold magenta")

    table.add_column("Metric", justify="right", style="purple", no_wrap=True)
    table.add_column("Value", justify="left", style="green")

    # File info
    table.add_row("File", str(filepath))

    # Model performance
    table.add_row("Total test games", f"{total_games:,}")
    table.add_row("Correct predictions", f"{correct_predictions:,}")
    table.add_row("Test accuracy", f"[bold]{accuracy:.1%}[/bold]")

    # Date range if available
    if 'GAME_DATE' in results.columns:
        dates = pd.to_datetime(results['GAME_DATE'])
        earliest = dates.min()
        latest = dates.max()
        span_days = (latest - earliest).days
        table.add_row("", "")  # Separator
        table.add_row("Earliest game", earliest.strftime('%Y-%m-%d'))
        table.add_row("Latest game", latest.strftime('%Y-%m-%d'))
        table.add_row("Date span", f"{span_days} days")

    # Win/Loss breakdown
    actual_wins = (results['ACTUAL_OUTCOME'] == 'W').sum()
    actual_losses = (results['ACTUAL_OUTCOME'] == 'L').sum()
    predicted_wins = (results['PREDICTED_OUTCOME'] == 'W').sum()
    predicted_losses = (results['PREDICTED_OUTCOME'] == 'L').sum()

    table.add_row("", "")  # Separator
    table.add_row("Actual Wins", f"{actual_wins:,}")
    table.add_row("Actual Losses", f"{actual_losses:,}")
    table.add_row("Predicted Wins", f"{predicted_wins:,}")
    table.add_row("Predicted Losses", f"{predicted_losses:,}")

    # Print the table once
    print()  # Empty line before table
    console.print(table)
    print()  # Empty line after table

    return filepath
