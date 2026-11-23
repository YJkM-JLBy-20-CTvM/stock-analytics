from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import ingest
import metrics
import moex
import report


def test_fetch_daily_reads_history_endpoint(monkeypatch):
    seen = {}

    def fake_paginate(path, block, **params):
        seen.update(path=path, block=block, params=params)
        return [
            {"SECID": "sber", "TRADEDATE": "2026-01-05", "CLOSE": "300,5", "OPEN": "299",
             "HIGH": "301", "LOW": "298", "VOLUME": "1 000", "VALUE": "300 500",
             "NUMTRADES": "10"},
            {"SECID": "GAZP", "TRADEDATE": "2026-01-05", "CLOSE": "120,1"},
        ]

    monkeypatch.setattr(moex, "_paginate", fake_paginate)
    rows = moex.fetch_daily(["SBER"], date(2026, 1, 1), date(2026, 1, 31))

    assert seen["path"] == moex.HISTORY_PATH
    assert seen["path"].startswith("/history/")
    assert seen["block"] == "history"
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "SBER"
    assert row["close"] == pytest.approx(300.5)
    assert row["volume"] == pytest.approx(1000.0)
    assert row["turnover"] == pytest.approx(300500.0)


def test_fetch_daily_falls_back_to_legal_close(monkeypatch):
    monkeypatch.setattr(moex, "_paginate", lambda *a, **k: [
        {"SECID": "SBER", "TRADEDATE": "2026-01-05", "CLOSE": None, "LEGALCLOSEPRICE": "120,1"},
    ])
    rows = moex.fetch_daily(["SBER"], date(2026, 1, 1), date(2026, 1, 31))
    assert rows[0]["close"] == pytest.approx(120.1)


def test_month_chunks_cover_period_without_gaps():
    chunks = ingest._month_chunks(date(2025, 11, 20), date(2026, 2, 10))
    assert chunks[0][0] == date(2025, 11, 20)
    assert chunks[-1][1] == date(2026, 2, 10)
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert next_start == prev_end + timedelta(days=1)
    assert all(start <= end for start, end in chunks)


def test_upsert_is_idempotent(tmp_path):
    con = ingest.connect(tmp_path / "test.duckdb")
    row = {"ticker": "SBER", "trade_date": date(2026, 1, 5), "open": 1.0, "high": 2.0,
           "low": 1.0, "close": 2.0, "volume": 100.0, "turnover": 200.0, "trades": 5.0}

    ingest._upsert(con, "prices", ingest.PRICE_COLUMNS, [row])
    row["close"] = 3.0
    ingest._upsert(con, "prices", ingest.PRICE_COLUMNS, [row])

    count, close = con.execute("SELECT count(*), max(close) FROM prices").fetchone()
    assert (count, close) == (1, 3.0)


def test_window_returns_do_not_backfill_short_history():
    idx = pd.bdate_range("2026-01-01", periods=3)
    closes = pd.DataFrame({"A": [10.0, 20.0, 30.0], "B": [np.nan, 5.0, 10.0]}, index=idx)

    norm = metrics.window_returns(closes, idx[0], idx[-1])

    assert np.isnan(norm["B"].iloc[0])
    assert norm["B"].iloc[-1] == pytest.approx(2.0)
    assert norm["A"].iloc[0] == pytest.approx(1.0)


def test_ranked_pairs_works_with_single_named_axis():
    matrix = pd.DataFrame([[1.0, 0.5, 0.2], [0.5, 1.0, 0.9], [0.2, 0.9, 1.0]],
                          index=["A", "B", "C"], columns=["A", "B", "C"])
    matrix.index.name = matrix.columns.name = "ticker"

    pairs = metrics.ranked_pairs(matrix, top=2)

    assert list(pairs.columns) == ["ticker_a", "ticker_b", "corr", "kind"]
    assert len(pairs) == 4
    assert pairs.iloc[0]["corr"] >= pairs.iloc[-1]["corr"]


def test_report_renders_dash_for_missing_values():
    idx = pd.Series([100.0, 110.0], index=pd.to_datetime(["2026-01-05", "2026-11-20"]))
    table = pd.DataFrame([{
        "ticker": "SBER", "short_name": "Сбербанк", "days": 250,
        "total_return": np.nan, "volatility": 0.2, "max_drawdown": -0.3,
        "beta": np.nan, "avg_turnover": np.nan,
        "first_day": pd.Timestamp("2026-01-05"), "last_day": pd.Timestamp("2026-11-20"),
    }])
    res = {
        "closes": pd.DataFrame(), "index": idx, "table": table, "corr": pd.DataFrame(),
        "pairs": pd.DataFrame(columns=["kind", "ticker_a", "ticker_b", "corr"]),
        "weights": pd.DataFrame(), "prices": pd.DataFrame(), "fx": pd.DataFrame(),
    }
    images = {k: "data:image/png;base64,AA" for k in
              ["returns", "normalized", "riskreturn", "weights", "turnover"]}

    html = report.build_html(res, idx.index[0].date(), idx.index[-1].date(), "IMOEX", images)

    assert "— млн" not in html
    assert "<td class='num'>—</td>" in html
