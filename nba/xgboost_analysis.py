import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


def find_top_features(data, target_col='WL', top_n=20, test_size=0.2, random_state=42, exclude_features=None):
    # create a copy of the data to avoid modifying the original data
    data_copy = data.copy()

    # convert the target column to numeric if it's string (e.g., 'W'/'L' -> 1/0)
    if data_copy[target_col].dtype == 'object':
        data_copy[target_col] = data_copy[target_col].map({'W': 1, 'L': 0})

    # select the feature columns
    # exclude the target column and any specified features
    if exclude_features is None:
        exclude_features = []
    feature_cols = [col for col in data_copy.select_dtypes(include=[np.number]).columns
                    if col != target_col and col not in exclude_features]

    # split the data into training and testing sets
    X = data_copy[feature_cols]
    y = data_copy[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

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

    return top_features, train_acc, test_acc
