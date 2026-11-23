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
# Официальный корневой сертификат Минцифры: без него Python не доверяет
# цепочке ISS на машинах с российским TLS-прокси. Проверка TLS остаётся включённой.
CA_BUNDLE = Path(__file__).with_name("certs") / "russian_trusted_root_ca.pem"


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if CA_BUNDLE.exists():
        context.load_verify_locations(cafile=str(CA_BUNDLE))
    return context

SHARES_PATH = "/engines/stock/markets/shares/boards/TQBR/securities"
TICKER_HISTORY_PATH = "/history/engines/stock/markets/shares/securities/{ticker}"
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
            with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
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
    previous: list[dict] | None = None
    while True:
        page = _block(_get(path, start=len(rows), limit=PAGE, **params), block)
        # ISS игнорирует start для справочника и отдаёт одну и ту же страницу:
        # повтор страницы — единственный надёжный признак конца
        if not page or page == previous:
            break
        rows.extend(page)
        previous = page
        if len(page) < PAGE:
            break
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
    """Дневная история одной бумаги: ISS не фильтрует by-борду по периоду, только по тикеру."""
    out = []
    for ticker in tickers:
        path = TICKER_HISTORY_PATH.format(ticker=ticker)
        rows = _paginate(path, "history", **{
            "from": date_from.isoformat(), "till": date_to.isoformat(),
            "history.columns": "SECID,TRADEDATE,CLOSE,LEGALCLOSEPRICE,OPEN,HIGH,LOW,VOLUME,VALUE,NUMTRADES",
        })
        secid = ticker.upper()
        for r in rows:
            if (r.get("SECID") or secid).upper() != secid:
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
