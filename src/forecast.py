"""
Forecast models + backtest.

Only 12 months of history, so anything with a yearly season (Holt-Winters,
SARIMA with a 52 week period) is off the table. You need at least two
cycles to estimate seasonality and we have less than one. So I stuck with
simple methods a planner could explain to a buyer:

  naive   - next week = this week
  ma4     - average of the last 4 weeks
  ma8     - average of the last 8 weeks
  ses     - simple exponential smoothing
  holt    - Holt's linear trend, damped (so it doesn't run off a cliff)

Backtest is rolling one step ahead: for each week in the holdout, fit on
everything before it, forecast that week, move on. That's how it would
actually be used (re-run every Monday).
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

warnings.filterwarnings("ignore")

METHODS = ["naive", "ma4", "ma8", "ses", "holt"]


def one_step(history, method):
    h = np.asarray(history, dtype=float)
    if method == "naive":
        return h[-1]
    if method == "ma4":
        return h[-4:].mean()
    if method == "ma8":
        return h[-8:].mean()
    if method == "ses":
        fit = SimpleExpSmoothing(h, initialization_method="estimated").fit()
        return max(fit.forecast(1)[0], 0)
    if method == "holt":
        fit = ExponentialSmoothing(h, trend="add", damped_trend=True,
                                   initialization_method="estimated").fit()
        return max(fit.forecast(1)[0], 0)
    raise ValueError(method)


def backtest(series, holdout=12):
    """Returns a frame of actual vs forecast for each method over the holdout."""
    y = series.values
    idx = series.index
    rows = []
    for t in range(len(y) - holdout, len(y)):
        hist = y[:t]
        row = {"week": idx[t], "actual": y[t]}
        for m in METHODS:
            row[m] = one_step(hist, m)
        rows.append(row)
    return pd.DataFrame(rows)


def wape(actual, fc):
    # weighted absolute percent error. Better than MAPE when some weeks are tiny
    denom = np.sum(np.abs(actual))
    return np.sum(np.abs(actual - fc)) / denom if denom else np.nan


def bias(actual, fc):
    # positive = over forecasting
    denom = np.sum(actual)
    return (np.sum(fc) - denom) / denom if denom else np.nan


def score(bt):
    out = {}
    for m in METHODS:
        e = bt[m] - bt["actual"]
        out[m] = {
            "wape": wape(bt["actual"], bt[m]),
            "bias": bias(bt["actual"], bt[m]),
            "rmse": float(np.sqrt(np.mean(e ** 2))),
        }
    return out
