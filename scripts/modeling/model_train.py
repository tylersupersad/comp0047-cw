import argparse
import json
import os
import pandas as pd
from data_utils import load_feature_dataset
from sklearn.model_selection import train_test_split
from models import (
    train_random_forest,
    train_xgboost,
    train_ridge,
    train_lasso,
    train_linear_regression
)
from evaluate import evaluate_predictions, log_results

def main():
    parser = argparse.ArgumentParser(description="train model to predict volatility_t+1")
    parser.add_argument("--input", required=True, help="path to feature csv")
    parser.add_argument("--target", required=True, choices=["volatility_t+1"], help="target to predict")
    parser.add_argument("--model", required=True, choices=["xgboost", "random_forest", "ridge", "lasso"], help="model to train")
    parser.add_argument("--sector", required=True, choices=["energy", "technology"], help="sector name")
    parser.add_argument("--output", required=True, help="path to save output JSON")
    parser.add_argument("--test_size", type=float, default=0.3, help="test split ratio")
    args = parser.parse_args()

    print(f"[✓] loading dataset: {args.input}")
    df = load_feature_dataset(args.input, args.target)

    features = [
        'return_t', 'volatility_t',
        'volume_t', 'market_cap',
        'return_t-1', 'return_t-2',
        'volatility_t-1', 'volatility_t-2',
        'volume_t-1', 'volume_t-2',
        'volume_rolling_mean_5', 'market_cap_rolling_mean_5'
    ]
    X = df[features]
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=False)

    # try to load tuned hyperparameters
    model_kwargs = {}
    tuning_file = f"tuning_{args.sector}_vol_rf.json"
    if args.model == "random_forest" and os.path.exists(tuning_file):
        with open(tuning_file) as f:
            model_kwargs = json.load(f).get("best_params", {})

    print(f"[→] training {args.model} model for {args.sector} sector")
    if args.model == "xgboost":
        model, importances = train_xgboost(X_train, y_train)
    elif args.model == "random_forest":
        model, importances = train_random_forest(X_train, y_train, **model_kwargs)
    elif args.model == "ridge":
        model, importances = train_ridge(X_train, y_train)
    elif args.model == "lasso":
        model, importances = train_lasso(X_train, y_train)
    else:
        model, importances = train_linear_regression(X_train, y_train)

    metrics, preds = evaluate_predictions(model, X_test, y_test)

    log_results(
        metrics,
        importances,
        args,
        model_name=args.model,
        y_test=y_test,
        y_pred=preds,
        model=model, 
        X_test=X_test
    )

if __name__ == "__main__":
    main()
