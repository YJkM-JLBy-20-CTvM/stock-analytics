"""Клиент ISS Мосбиржи: справочник бумаг, дневные котировки, состав и история индексов.

ISS отдаёт данные только в формате, который выбирается расширением в пути
(`/path.json`), а не заголовком Accept — на это легко наткнуться.
Числа приходят строками с запятой и пробелами-разделителями.
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

BASE = "https://iss.moex.com/iss"
SHARES_PATH = "/engines/stock/markets/shares/boards/TQBR/securities"
HISTORY_PATH = "/history" + SHARES_PATH
INDEX_PATH = "/history/engines/stock/markets/index/securities/{index}"
WEIGHTS_PATH = "/statistics/engines/stock/markets/index/analytics/{index}"
PAGE = 100


class MoexError(RuntimeError):
    pass


def _get(path: str, **params) -> dict:
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    url = f"{BASE}{path}.json?{query}"
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "moex-market-report/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # ISS периодически роняет TLS-цепочку и рвёт соединение: ждём дольше,
            # но итог один — MoexError, который ingest переживает по чанкам
            last_error = exc
            time.sleep(min(2 ** attempt, 25))
    raise MoexError(f"{url}: {last_error}")


def _block(payload: dict, name: str) -> list[dict]:
    block = payload.get(name) or {}
    columns = block.get("columns") or []
    return [dict(zip(columns, row)) for row in block.get("data") or []]


def _paginate(path: str, block: str, **params) -> list[dict]:
    rows: list[dict] = []
    empty_pages = 0
    while empty_pages < 2:
        chunk = _block(_get(path, start=len(rows), limit=PAGE, **params), block)
        if not chunk:
            empty_pages += 1
            continue
        empty_pages = 0
        rows.extend(chunk)
    return rows


def num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\xa0", "").replace(" ", "").replace(",", ".")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_universe() -> list[dict]:
    """Все акции, торгующиеся в основном режиме TQBR (ETF и паи сюда же)."""
    rows = _paginate(SHARES_PATH, "securities",
                     securities_columns="SECID,SHORTNAME,SECNAME,LOTSIZE")
    return [
        {"ticker": r["SECID"].upper(), "short_name": r.get("SHORTNAME"),
         "full_name": r.get("SECNAME"), "lot_size": num(r.get("LOTSIZE"))}
        for r in rows if r.get("SECID")
    ]


def fetch_daily(tickers: list[str], date_from: date, date_to: date) -> list[dict]:
    wanted = {t.upper() for t in tickers}
    rows = _paginate(HISTORY_PATH, "history", **{
        "from": date_from.isoformat(), "till": date_to.isoformat(),
        "history.columns": "SECID,TRADEDATE,CLOSE,LEGALCLOSEPRICE,OPEN,HIGH,LOW,VOLUME,VALUE,NUMTRADES",
    })
    out = []
    for r in rows:
        secid = (r.get("SECID") or "").upper()
        if secid not in wanted:
            continue
        close = num(r.get("CLOSE"))
        if close is None:
            close = num(r.get("LEGALCLOSEPRICE"))
        out.append({
            "ticker": secid, "trade_date": r.get("TRADEDATE"), "close": close,
            "open": num(r.get("OPEN")), "high": num(r.get("HIGH")), "low": num(r.get("LOW")),
            "volume": num(r.get("VOLUME")), "turnover": num(r.get("VALUE")),
            "trades": num(r.get("NUMTRADES")),
        })
    return out


def fetch_index_weights(index: str, on: date) -> list[dict]:
    rows = _paginate(WEIGHTS_PATH.format(index=index), "analytics", date=on.isoformat())
    return [
        {"index_id": index, "weight_date": r.get("tradedate"),
         "ticker": r["ticker"].upper(), "weight": num(r.get("weight"))}
        for r in rows if r.get("ticker") and r["ticker"].upper() != index
    ]


def fetch_index_history(index: str, date_from: date, date_to: date) -> list[dict]:
    rows = _paginate(INDEX_PATH.format(index=index), "history", **{
        "from": date_from.isoformat(), "till": date_to.isoformat(),
        "history.columns": "TRADEDATE,CLOSE,LEGALCLOSEPRICE",
    })
    out = []
    for r in rows:
        close = num(r.get("CLOSE"))
        if close is None:
            close = num(r.get("LEGALCLOSEPRICE"))
        if close is not None:
            out.append({"index_id": index, "trade_date": r.get("TRADEDATE"), "close": close})
    return out
