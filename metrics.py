"""Расчёт метрик из загруженных данных: доходности, риск, корреляции, ликвидность.

Все функции чистые и работают с pandas-объектами, чтобы их можно было тестировать
без сети и без базы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def pivot_closes(prices: pd.DataFrame) -> pd.DataFrame:
    """Широкий формат: индекс — даты, столбцы — тикеры, только реальные торги."""
    if prices.empty:
        return pd.DataFrame()
    wide = (prices.pivot_table(index="trade_date", columns="ticker", values="close", aggfunc="last")
            .sort_index())
    wide.index = pd.to_datetime(wide.index)
    return wide.dropna(how="all")


def window_returns(closes: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Нормированные к 1 ряды внутри окна — база для графиков «что было бы с 1 ₽».

    База у каждой бумаги — её первая реальная цена: бумага, вышедшая позже,
    просто начинается позже, а не получает нарисованный старт.
    """
    sub = closes.loc[(closes.index >= start) & (closes.index <= end)]
    if sub.empty:
        return sub
    base = sub.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    return sub.divide(base)


def total_return(closes: pd.DataFrame) -> pd.Series:
    """Доходность за окно, считается по первой и последней непустой цене."""
    return closes.apply(lambda s: s.dropna().iloc[-1] / s.dropna().iloc[0] - 1
                        if s.notna().sum() >= 2 else np.nan)


def daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    return closes.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def annualized_volatility(closes: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.Series:
    return daily_returns(closes).std(ddof=1) * np.sqrt(periods)


def max_drawdown(closes: pd.DataFrame) -> pd.Series:
    """Худшая просадка от локального максимума, отрицательное число."""
    clean = closes.apply(lambda s: s.ffill())
    peak = clean.cummax()
    return (clean / peak - 1).min()


def beta(closes: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    """Бета дневных доходностей бумаг к дневным доходностям индекса.

    `benchmark` — уже дневные доходности индекса: после смены базы ценностей
    отличить возвраты от цен по величине ненадёжно, поэтому пересчёт из цен не делаем.
    """
    rets = daily_returns(closes)
    bench = benchmark.reindex(rets.index)

    def _one(col: pd.Series) -> float:
        # ковариация и дисперсия — на одном и том же наборе общих дней,
        # иначе первый NaN доходности занижает бету
        mask = col.notna() & bench.notna()
        if mask.sum() < 2:
            return np.nan
        x, y = bench[mask], col[mask]
        variance = x.var(ddof=1)
        return y.cov(x, ddof=1) / variance if variance else np.nan

    return rets.apply(_one)


def inflation_adjusted(total: pd.Series, inflation: float) -> pd.Series:
    """Реальная доходность по Фишеру, если известна накопленная инфляция за период."""
    return (1 + total) / (1 + inflation) - 1


def correlation_matrix(closes: pd.DataFrame, min_obs: int = 60) -> pd.DataFrame:
    """Попарная корреляция дневных доходностей на пересечении дат.

    `min_obs` — сколько общих торговых дней нужно, чтобы считать корреляцию
    осмысленной; иначе NaN.
    """
    rets = daily_returns(closes)
    matrix = rets.corr(min_periods=min_obs)
    return matrix


def ranked_pairs(matrix: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Верхняя и нижняя часть корреляционной матрицы без повторов и диагонали."""
    upper = matrix.where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    # у матрицы обе оси называются ticker, и reset_index падает на дубле имени
    stacked = (upper.rename_axis(index="ticker_a", columns="ticker_b")
               .stack().dropna().rename("corr").reset_index())
    ordered = stacked.sort_values("corr")
    low = ordered.head(top).assign(kind="самые независимые")
    high = ordered.tail(top).iloc[::-1].assign(kind="самые связанные")
    return pd.concat([high, low], ignore_index=True)


def liquidity_table(prices: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Средний дневной оборот в рублях и число сделок — прокси ликвидности."""
    if prices.empty:
        return pd.DataFrame(columns=["avg_turnover", "avg_trades", "days"])
    agg = prices.groupby("ticker").agg(
        avg_turnover=("turnover", "mean"), avg_trades=("trades", "mean"),
        days=("trade_date", "nunique"))
    return agg.join(meta.set_index("ticker")[["short_name"]], how="left")


def concentration(weights: pd.DataFrame, index_id: str, on: pd.Timestamp) -> pd.DataFrame:
    """Топ по весам на ближайшую к `on` дату пересчёта состава."""
    sub = weights[weights["index_id"] == index_id].copy()
    if sub.empty:
        return sub
    sub["weight_date"] = pd.to_datetime(sub["weight_date"])
    date = sub["weight_date"].max() if on is None else sub["weight_date"].iloc[
        (sub["weight_date"] - on).abs().argsort().iloc[0]]
    chosen = sub[sub["weight_date"] == date].sort_values("weight", ascending=False)
    chosen["weight_share"] = chosen["weight"] / chosen["weight"].sum()
    return chosen
