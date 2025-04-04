import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# train random forest model
def train_random_forest(X, y, **kwargs):
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs)
    model.fit(X, y)
    importances = dict(zip(X.columns, model.feature_importances_))
    return model, importances

# train xgboost model
def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    return model, importances

# train linear regression (baseline)
def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    importances = pd.Series(model.coef_, index=X_train.columns)
    return model, importances

def train_ridge(X, y):
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model, dict(zip(X.columns, model.coef_))

def train_lasso(X, y):
    model = Lasso(alpha=0.01)
    model.fit(X, y)
    return model, dict(zip(X.columns, model.coef_))