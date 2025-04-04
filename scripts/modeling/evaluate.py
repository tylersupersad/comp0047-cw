import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
import shap
import os

# evaluate predictions on test set
def evaluate_predictions(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds)
    }
    return metrics, preds

# compute permutation importances
def compute_permutation_importance(model, X_test, y_test):
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    return dict(zip(X_test.columns, result.importances_mean))

# run statistical tests on residuals
def residual_diagnostics(y_true, y_pred):
    residuals = y_true - y_pred
    _, p_shapiro = shapiro(residuals)
    lb_test = acorr_ljungbox(residuals, lags=[10])
    return {
        "shapiro_p": float(p_shapiro),
        "ljung_box_p": float(lb_test['lb_pvalue'].values[0])
    }

# visualize prediction vs actual
def plot_predicted_vs_actual(y_true, y_pred, sector, model, output_dir):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual ({sector}, {model})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{sector}_{model}_predicted_vs_actual.png")
    plt.savefig(path)
    plt.close()

# visualize residuals
def plot_residuals(y_true, y_pred, sector, model, output_dir):
    residuals = y_true - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, bins=50, kde=True)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel("Residual")
    plt.title(f"Residual Distribution ({sector}, {model})")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{sector}_{model}_residuals.png")
    plt.savefig(path)
    plt.close()

# shap importance plot
def plot_shap_importance(model, X_test, sector, model_name, out_dir):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    fname = os.path.join(out_dir, f"{sector}_{model_name}_shap_summary.png")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"[✓] SHAP summary plot saved to {fname}")

# log results to json and plots
def log_results(metrics, importances, args, model_name, y_test=None, y_pred=None, model=None, X_test=None):
    metrics = {k: float(v) for k, v in metrics.items()}
    importances = dict(importances) if not isinstance(importances, dict) else importances
    importances = {k: float(v) for k, v in importances.items()}

    results = {
        "model": model_name,
        "sector": args.sector,
        "target": args.target,
        "metrics": metrics,
        "importances": importances
    }

    if y_test is not None and y_pred is not None:
        diagnostics = residual_diagnostics(y_test, y_pred)
        diagnostics = {k: float(v) for k, v in diagnostics.items()}
        results["diagnostics"] = diagnostics

    out_dir = os.path.dirname(args.output)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    if y_test is not None and y_pred is not None:
        plot_predicted_vs_actual(y_test, y_pred, args.sector, model_name, out_dir)
        plot_residuals(y_test, y_pred, args.sector, model_name, out_dir)

    if model_name in ["random_forest", "xgboost"] and model is not None and X_test is not None:
        try:
            plot_shap_importance(model, X_test, args.sector, model_name, out_dir)
        except Exception as e:
            print(f"[!] SHAP plot skipped: {e}")

    print("\n[✓] Evaluation saved with diagnostics and visualizations.")