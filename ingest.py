"""Загрузка данных Мосбиржи в DuckDB: котировки, состав индексов, история индексов и ЦБ."""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import duckdb
import pandas as pd

import cbr
import moex

DEFAULT_INDEXES = ["IMOEX", "MOEXBC"]
DEFAULT_DB = Path(__file__).resolve().parent / "data" / "market.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS securities (
    ticker      VARCHAR PRIMARY KEY,
    short_name  VARCHAR,
    full_name   VARCHAR,
    lot_size    DOUBLE
);
CREATE TABLE IF NOT EXISTS prices (
    ticker      VARCHAR,
    trade_date  DATE,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    turnover    DOUBLE,
    trades      DOUBLE,
    loaded_at   TIMESTAMP DEFAULT now(),
    PRIMARY KEY (ticker, trade_date)
);
CREATE TABLE IF NOT EXISTS index_weights (
    index_id    VARCHAR,
    weight_date DATE,
    ticker      VARCHAR,
    weight      DOUBLE,
    loaded_at   TIMESTAMP DEFAULT now(),
    PRIMARY KEY (index_id, weight_date, ticker)
);
CREATE TABLE IF NOT EXISTS index_prices (
    index_id    VARCHAR,
    trade_date  DATE,
    close       DOUBLE,
    loaded_at   TIMESTAMP DEFAULT now(),
    PRIMARY KEY (index_id, trade_date)
);
CREATE TABLE IF NOT EXISTS fx_rates (
    currency    VARCHAR,
    rate_date   DATE,
    rate        DOUBLE,
    source      VARCHAR,
    PRIMARY KEY (currency, rate_date)
);
CREATE TABLE IF NOT EXISTS load_log (
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    status      VARCHAR,
    details     VARCHAR
);
"""


def connect(path: Path) -> duckdb.DuckDBPyConnection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA)
    return con


def _upsert(con, table: str, columns: list[str], rows: list[dict]) -> int:
    """Одна временная таблица + MERGE вместо executemany: executemany строчными
    записями на десятках тысяч строк занимает минуты."""
    if not rows:
        return 0
    frame = pd.DataFrame([{c: r.get(c) for c in columns} for r in rows], columns=columns)
    tmp = f"stg_{table}"
    con.register("_stg", frame)
    con.execute(f"CREATE OR REPLACE TEMP TABLE {tmp} AS SELECT * FROM _stg")
    keys = {
        "securities": ["ticker"],
        "prices": ["ticker", "trade_date"],
        "index_prices": ["index_id", "trade_date"],
        "index_weights": ["index_id", "weight_date", "ticker"],
        "fx_rates": ["currency", "rate_date"],
    }[table]
    on = " AND ".join(f"t.{k} = s.{k}" for k in keys)
    con.execute(f"DELETE FROM {table} AS t USING {tmp} AS s WHERE {on}")
    con.execute(f"INSERT INTO {table} ({', '.join(columns)}) SELECT {', '.join(columns)} FROM {tmp}")
    con.execute(f"DROP TABLE {tmp}")
    return len(frame)


PRICE_COLUMNS = ["ticker", "trade_date", "open", "high", "low", "close",
                 "volume", "turnover", "trades"]


def _month_chunks(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    """Режем период по календарным месяцам: каждый чанк сохраняется отдельно."""
    chunks: list[tuple[dt.date, dt.date]] = []
    cur = start
    while cur <= end:
        next_month = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        chunk_end = min(end, next_month - dt.timedelta(days=1))
        chunks.append((cur, chunk_end))
        cur = chunk_end + dt.timedelta(days=1)
    return chunks


def run(date_from: dt.date, date_to: dt.date, indexes: list[str], db_path: Path) -> dict:
    con = connect(db_path)
    started = dt.datetime.now()
    summary: dict[str, object] = {}
    con.execute("INSERT INTO load_log (started_at, status) VALUES (?, 'running')", [started])
    try:
        universe = moex.fetch_universe()
        summary["securities"] = _upsert(con, "securities",
                                        ["ticker", "short_name", "full_name", "lot_size"], universe)

        tickers = [s["ticker"] for s in universe]
        prices, missed = 0, []
        for chunk_from, chunk_to in _month_chunks(date_from, date_to):
            try:
                rows = moex.fetch_daily(tickers, chunk_from, chunk_to)
            except moex.MoexError as exc:
                missed.append(f"{chunk_from}..{chunk_to}: {exc}"[:300])
                print(f"[warn] чанк {chunk_from}..{chunk_to} пропущен", file=sys.stderr, flush=True)
                continue
            prices += _upsert(con, "prices", PRICE_COLUMNS, rows)
            print(f"[ok] {chunk_from}..{chunk_to}: {prices} строк", file=sys.stderr, flush=True)
        summary["prices"] = prices
        if missed:
            summary["missed_chunks"] = missed

        index_prices, weights = [], []
        for idx in indexes:
            index_prices.extend(moex.fetch_index_history(idx, date_from, date_to))
            for weight_date in (date_to, date_from):
                try:
                    weights.extend(moex.fetch_index_weights(idx, weight_date))
                    break
                except moex.MoexError:
                    continue
        summary["index_prices"] = _upsert(con, "index_prices",
                                          ["index_id", "trade_date", "close"], index_prices)
        summary["index_weights"] = _upsert(con, "index_weights",
                                           ["index_id", "weight_date", "ticker", "weight"], weights)

        fx = []
        for on in _business_days(date_to - dt.timedelta(days=10), date_to):
            try:
                rate = cbr.fetch_usd_rate(on)
            except Exception:
                continue
            if rate:
                fx.append({"currency": "USD", "rate_date": on, "rate": rate, "source": "cbr"})
        summary["fx_rates"] = _upsert(con, "fx_rates",
                                      ["currency", "rate_date", "rate", "source"], fx)

        con.execute(
            "UPDATE load_log SET finished_at=now(), status='ok', details=? WHERE started_at=?",
            [str(summary)[:2000], started])
        return summary
    except Exception as exc:
        con.execute(
            "UPDATE load_log SET finished_at=now(), status='failed', details=? WHERE started_at=?",
            [str(exc)[:500], started])
        raise
    finally:
        con.close()


def _business_days(start: dt.date, end: dt.date) -> list[dt.date]:
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    return [d for d in days if d.weekday() < 5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать данные Мосбиржи в DuckDB")
    parser.add_argument("--from", dest="date_from",
                        default=(dt.date.today() - dt.timedelta(days=730)).isoformat())
    parser.add_argument("--to", dest="date_to", default=dt.date.today().isoformat())
    parser.add_argument("--indexes", default=",".join(DEFAULT_INDEXES))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()
    indexes = [i.strip().upper() for i in args.indexes.split(",") if i.strip()]
    summary = run(dt.date.fromisoformat(args.date_from), dt.date.fromisoformat(args.date_to),
                  indexes, Path(args.db))
    for key, value in summary.items():
        print(f"{key:14} {value}", flush=True)


if __name__ == "__main__":
    main()
