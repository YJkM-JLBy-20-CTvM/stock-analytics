from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import metrics
import moex


def _frame(data: dict[str, list[float]], start="2026-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(next(iter(data.values()))))
    return pd.DataFrame(data, index=idx)


def test_num_parses_iss_formats():
    assert moex.num("1 234 567,89") == pytest.approx(1234567.89)
    assert moex.num("0,26") == pytest.approx(0.26)
    assert moex.num(None) is None
    assert moex.num("") is None
    assert moex.num(42) == 42.0


def test_total_return_uses_first_and_last_real_price():
    closes = _frame({"A": [100, 110, 120, 130], "B": [50, 40, 30, 20]})
    total = metrics.total_return(closes)
    assert total["A"] == pytest.approx(0.3)
    assert total["B"] == pytest.approx(-0.6)


def test_max_drawdown_is_worst_peak_to_trough():
    closes = _frame({"A": [100, 120, 60, 90]})
    assert metrics.max_drawdown(closes)["A"] == pytest.approx(-0.5)


def test_volatility_zero_for_flat_series():
    closes = _frame({"A": [10] * 30})
    assert metrics.annualized_volatility(closes)["A"] == pytest.approx(0.0)


def test_beta_of_benchmark_against_itself_is_one():
    rng = np.random.default_rng(7)
    closes = _frame({"A": (1 + rng.normal(0, 0.01, 120)).cumprod() * 100})
    bench = metrics.daily_returns(closes).iloc[:, 0]
    assert metrics.beta(closes, bench)["A"] == pytest.approx(1.0)


def test_beta_is_scale_invariant_for_returns():
    # шум в 1e-6: волатильность индекса ~1%, иначе шум съест бета
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0, 0.01, 100), index=pd.bdate_range("2026-01-01", periods=100))
    stock_returns = bench * 2 + rng.normal(0, 1e-6, 100)
    closes = (1 + stock_returns).cumprod().to_frame("S")
    assert metrics.beta(closes, bench)["S"] == pytest.approx(2.0, abs=0.02)


def test_correlation_ignores_non_overlapping_days():
    a = _frame({"A": [1, 2, 3, 4, 5, 6, 7, 8]})
    b = a.rename(columns={"A": "B"})
    b.iloc[0] = np.nan
    corr = metrics.correlation_matrix(pd.concat([a, b], axis=1), min_obs=3)
    assert corr.loc["A", "B"] == pytest.approx(1.0)


def test_correlation_needs_min_obs():
    closes = _frame({"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]})
    matrix = metrics.correlation_matrix(closes, min_obs=60)
    assert np.isnan(matrix.loc["A", "B"])


def test_normalized_window_starts_at_one():
    closes = _frame({"A": [10, 20, 30]})
    norm = metrics.window_returns(closes, closes.index[0], closes.index[-1])
    assert norm.iloc[0, 0] == pytest.approx(1.0)
    assert norm.iloc[-1, 0] == pytest.approx(3.0)


def test_inflation_adjusted_amounts_to_fisher():
    total = pd.Series({"A": 0.20})
    real = metrics.inflation_adjusted(total, 0.10)
    assert real["A"] == pytest.approx(1.2 / 1.1 - 1)


def test_weight_shares_are_normalized():
    weights = pd.DataFrame({
        "index_id": ["IMOEX", "IMOEX", "IMOEX"],
        "weight_date": ["2026-11-20"] * 3,
        "ticker": ["A", "B", "C"],
        "weight": [10.0, 30.0, 60.0],
    })
    top = metrics.concentration(weights, "IMOEX", None)
    assert top["weight_share"].sum() == pytest.approx(1.0)
    assert top.iloc[0]["ticker"] == "C"
