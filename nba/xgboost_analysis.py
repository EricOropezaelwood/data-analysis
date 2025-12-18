import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from preprocessing import get_feature_columns


def find_top_features(data, target_col='WL', top_n=20, test_size=0.2, random_state=42,
                      exclude_features=None):
    # create a copy of the data to avoid modifying the original data
    data_copy = data.copy()

    # convert the target column to numeric if it's string (e.g., 'W'/'L' -> 1/0)
    if data_copy[target_col].dtype == 'object':
        data_copy[target_col] = data_copy[target_col].map({'W': 1, 'L': 0})

    # Use only pre-game features (to avoid data leakage, saw this on early testing)
    feature_cols = get_feature_columns(data_copy)
    print(f"✓ Using pre-game features only (no data leakage)")


    # split the data into training and testing sets
    X = data_copy[feature_cols]
    y = data_copy[target_col]

    # Always use time-based split: use most recent games as test set
    #  Note: Just makes sense to me given the nature of NBA seasons
    if 'GAME_DATE' not in data_copy.columns:
        raise ValueError("GAME_DATE column required for time-based split")

    print(f"Using time-based split (most recent {test_size:.0%} of games as test set)")
    data_copy['GAME_DATE'] = pd.to_datetime(data_copy['GAME_DATE'])
    sorted_data = data_copy.sort_values('GAME_DATE')
    split_idx = int(len(sorted_data) * (1 - test_size))

    train_indices = sorted_data.index[:split_idx]
    test_indices = sorted_data.index[split_idx:]

    X_train = X.loc[train_indices]
    X_test = X.loc[test_indices]
    y_train = y.loc[train_indices]
    y_test = y.loc[test_indices]

    print(f"  Train: {len(X_train)} games")
    print(f"  Test: {len(X_test)} games (most recent)")

    model = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        enable_categorical=False
    )
    model.fit(X_train, y_train)

    importance = model.get_booster().get_score(importance_type='gain')

    print(f"XGBoost used {len(importance)}/{len(feature_cols)} features in the model")

    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_features = dict(sorted_features[:top_n])

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    # Get predictions and probabilities for test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    return top_features, train_acc, test_acc, X_test, y_test, y_pred, y_pred_proba
