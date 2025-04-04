import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# run grid search using time-aware cross-validation
def tune_model(model_type, X_train, y_train):
    if model_type == "xgboost":
        model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
            "gamma": [0, 1],
            "reg_alpha": [0, 0.5],
            "reg_lambda": [1]
        }
    elif model_type == "random_forest":
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        param_grid = {
            "n_estimators": [100, 300],
            "max_depth": [10, 15],
            "min_samples_leaf": [3, 5],
            "max_features": ["sqrt"]
        }

    # use time-aware splitting
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        model,
        param_grid,
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_

# compute performance metrics on test set
def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds)
    }

def main():
    parser = argparse.ArgumentParser(description="hyperparameter tuning for xgboost or random forest")
    parser.add_argument("--input", required=True, help="path to feature csv")
    parser.add_argument("--target", required=True, choices=["return_t+1", "volatility_t+1"], help="prediction target")
    parser.add_argument("--model", required=True, choices=["xgboost", "random_forest"], help="model type")
    parser.add_argument("--test_size", type=float, default=0.2, help="test set proportion")
    parser.add_argument("--output", help="optional path to save results json")
    args = parser.parse_args()

    # load dataset and drop rows with missing target
    df = pd.read_csv(args.input, parse_dates=["date"])
    df = df.dropna(subset=[args.target])

    # define input features
    features = [
        "return_t", "volatility_t",
        "volume_t", "market_cap",
        "return_t-1", "return_t-2",
        "volatility_t-1", "volatility_t-2",
        "volume_t-1", "volume_t-2",
        "volume_rolling_mean_5", "market_cap_rolling_mean_5"
    ]
    X = df[features]
    y = df[args.target]

    # use chronological split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, shuffle=False
    )

    print(f"[→] tuning {args.model} to predict {args.target}...")
    model, best_params = tune_model(args.model, X_train, y_train)
    metrics = evaluate(model, X_test, y_test)

    print("\n[✓] best parameters:")
    for k, v in best_params.items():
        print(f"   {k}: {v}")
    print("\n[✓] evaluation on test set:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # optionally export results to file
    if args.output:
        result = {
            "model": args.model,
            "target": args.target,
            "best_params": best_params,
            "metrics": metrics
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
