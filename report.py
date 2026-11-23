"""Сборка метрик и автономного HTML-отчёта из данных DuckDB."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mtick  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import metrics  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 110, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "semibold",
})


def load(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    def q(sql: str) -> pd.DataFrame:
        return con.execute(sql).df()

    return {
        "prices": q("SELECT * FROM prices ORDER BY trade_date"),
        "securities": q("SELECT * FROM securities"),
        "index_prices": q("SELECT * FROM index_prices ORDER BY trade_date"),
        "weights": q("SELECT * FROM index_weights"),
        "fx": q("SELECT * FROM fx_rates ORDER BY rate_date"),
    }


def compute(data: dict[str, pd.DataFrame], index_id: str = "IMOEX") -> dict:
    prices, meta = data["prices"], data["securities"]
    closes = metrics.pivot_closes(prices)
    if closes.empty:
        raise SystemExit("В базе нет цен — сначала выполните ingest.py")

    idx = data["index_prices"]
    idx = idx[idx["index_id"] == index_id].sort_values("trade_date")
    if idx.empty:
        raise SystemExit(f"В базе нет истории индекса {index_id}")
    idx_close = pd.Series(idx["close"].to_numpy(),
                          index=pd.to_datetime(idx["trade_date"].to_numpy()), name=index_id)

    total = metrics.total_return(closes)
    vol = metrics.annualized_volatility(closes)
    dd = metrics.max_drawdown(closes)
    bet = metrics.beta(closes, metrics.daily_returns(idx_close.to_frame()).iloc[:, 0])
    liq = metrics.liquidity_table(prices, meta)
    corr = metrics.correlation_matrix(closes)
    weights = (metrics.concentration(data["weights"], index_id, None)
               if not data["weights"].empty else pd.DataFrame())

    names = meta.set_index("ticker")["short_name"]
    table = pd.DataFrame({
        "ticker": closes.columns,
        "short_name": closes.columns.map(names),
        "days": closes.notna().sum().to_numpy(),
        "total_return": total.reindex(closes.columns).to_numpy(),
        "volatility": vol.reindex(closes.columns).to_numpy(),
        "max_drawdown": dd.reindex(closes.columns).to_numpy(),
        "beta": bet.reindex(closes.columns).to_numpy(),
        "avg_turnover": liq.reindex(closes.columns)["avg_turnover"].to_numpy(),
        "first_day": [closes[t].first_valid_index() for t in closes.columns],
        "last_day": [closes[t].last_valid_index() for t in closes.columns],
    })
    return {"closes": closes, "index": idx_close, "table": table, "corr": corr,
            "pairs": metrics.ranked_pairs(corr), "weights": weights,
            "prices": prices, "fx": data["fx"]}


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _blank(message: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 1.4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#718096")
    return _png(fig)


def chart_returns(table: pd.DataFrame, idx_ret: float, index_id: str, top: int = 20) -> str:
    sub = table.dropna(subset=["total_return"]).nlargest(top, "total_return")
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(sub) + 1.2))
    ax.barh(sub["ticker"], sub["total_return"],
            color=["#1f8a4c" if r >= 0 else "#c0392b" for r in sub["total_return"]])
    ax.axvline(idx_ret, color="#333", ls="--", lw=1, label=f"{index_id}: {signed_pct(idx_ret)}")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.invert_yaxis()
    ax.set_title(f"Доходность за период, топ-{len(sub)}")
    ax.legend(frameon=False, loc="lower right")
    return _png(fig)


def chart_normalized(res: dict, index_id: str, top: int = 8) -> str:
    table, closes = res["table"], res["closes"]
    leaders = list(table.dropna(subset=["total_return"]).nlargest(top, "total_return")["ticker"])
    norm = metrics.window_returns(closes[leaders], closes.index[0], closes.index[-1])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    norm.plot(ax=ax, lw=1.2, cmap="tab10")
    idx = res["index"]
    ax.plot(idx.index, idx / idx.iloc[0], color="black", ls="--", lw=1.8, label=index_id)
    ax.set_title(f"1 ₽ в начале периода — сколько к концу (пунктир: {index_id})")
    ax.set_ylabel("рост вложенной суммы")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    return _png(fig)


def chart_risk_return(table: pd.DataFrame) -> str:
    sub = table.dropna(subset=["total_return", "volatility"])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.scatter(sub["volatility"], sub["total_return"], s=28, alpha=0.75, color="#2b6cb0")
    for _, r in sub.nlargest(8, "total_return").iterrows():
        ax.annotate(r["ticker"], (r["volatility"], r["total_return"]), fontsize=8,
                    xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="#888", lw=0.8)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Годовая волатильность")
    ax.set_ylabel("Доходность за период")
    ax.set_title("Риск и доходность: точка = бумага")
    return _png(fig)


def chart_weights(weights: pd.DataFrame) -> str:
    sub = weights.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(sub) + 1))
    ax.barh(sub["ticker"], sub["weight_share"], color="#6b46c1")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    date = pd.to_datetime(weights["weight_date"].iloc[0]).date()
    ax.set_title(f"Топ-{len(sub)} весов в индексе на {date}")
    return _png(fig)


def chart_market_turnover(res: dict) -> str:
    daily = res["prices"].groupby("trade_date")["turnover"].sum() / 1e9
    fig, ax = plt.subplots(figsize=(9, 3.2))
    daily.plot(ax=ax, color="#2f855a", lw=1)
    ax.set_title("Суммарный оборот акций TQBR, млрд ₽/день")
    ax.set_xlabel("")
    return _png(fig)


def fx_note(fx: pd.DataFrame) -> str:
    if fx.empty:
        return "Курс ЦБ за период не загружен."
    first, last = fx.iloc[0], fx.iloc[-1]
    return (f"Доллар по ЦБ за последние {len(fx)} рабочих дней: "
            f"{first['rate']:.2f} ₽ ({first['rate_date'].date()}) → "
            f"{last['rate']:.2f} ₽ ({last['rate_date'].date()}), "
            f"{last['rate'] / first['rate'] - 1:+.1%}.")


def pct(value) -> str:
    return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:+.1%}"


def signed_pct(value: float) -> str:
    # f"-0.0%" выглядит как ошибка, хотя это ноль: порог — половина шага округления
    return f"{0.0 if abs(value) < 5e-4 else value:+.1%}"


def num(value, fmt: str) -> str:
    return "—" if value is None or (isinstance(value, float) and np.isnan(value)) else format(value, fmt)


def build_html(res: dict, date_from, date_to, index_id: str, images: dict[str, str]) -> str:
    import html

    table, idx = res["table"], res["index"]
    idx_ret = float(idx.iloc[-1] / idx.iloc[0] - 1)
    # сравнивать с индексом можно только бумаги с сопоставимой историей
    comparable = table[table["days"] >= 60]
    market_total = float(comparable["total_return"].mean()) if not comparable.empty else np.nan
    above = int((comparable["total_return"] > idx_ret).sum())
    median_dd = float(comparable["max_drawdown"].median()) if not comparable.empty else np.nan
    swing = float(idx.max() / idx.min() - 1)

    def table_rows(df: pd.DataFrame) -> str:
        cells = []
        for _, r in df.iterrows():
            turnover = num(r["avg_turnover"] / 1e6, ",.1f")
            turnover = "—" if turnover == "—" else turnover + " млн"
            up = pd.notna(r["total_return"]) and r["total_return"] >= 0
            cells.append(
                "<tr>"
                f"<td>{html.escape(str(r['ticker']))}</td>"
                f"<td>{html.escape(str(r['short_name']))}</td>"
                f"<td class='num'>{int(r['days'])}</td>"
                f"<td class='num {'pos' if up else 'neg'}'>{pct(r['total_return'])}</td>"
                f"<td class='num'>{pct(r['volatility'])}</td>"
                f"<td class='num neg'>{pct(r['max_drawdown'])}</td>"
                f"<td class='num'>{num(r['beta'], '.2f')}</td>"
                f"<td class='num'>{turnover}</td>"
                "</tr>")
        return "".join(cells)

    header = ("<thead><tr><th>Тикер</th><th>Название</th><th>Дней</th><th>Доходность</th>"
              "<th>Волат.</th><th>Макс. просадка</th><th>Бета</th><th>Оборот/день</th></tr></thead>")
    top = table.dropna(subset=["total_return"]).nlargest(12, "total_return")
    volatile = table.sort_values("volatility", ascending=False).head(8)
    pairs_html = "".join(
        f"<tr><td>{p['kind']}</td><td>{p['ticker_a']}</td><td>{p['ticker_b']}</td>"
        f"<td class='num'>{p['corr']:.2f}</td></tr>"
        for _, p in res["pairs"].iterrows())
    short_history = "".join(
        f"<tr><td>{html.escape(row['ticker'])}</td><td>{html.escape(str(row['short_name']))}</td>"
        f"<td>{row['first_day'].date()}</td><td>{row['last_day'].date()}</td>"
        f"<td>{int(row['days'])}</td></tr>"
        for _, row in table.sort_values("days").head(10).iterrows())
    weights_note = "" if not res["weights"].empty else \
        "<p class='note'>Состав индекса на дату в базе отсутствует, блок пуст — это честнее, чем подставлять старые веса.</p>"

    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Обзор рынка акций Мосбиржи {date_from} — {date_to}</title>
<style>
 body {{ font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; margin:0; background:#f7f8fa; color:#1a202c; }}
 main {{ max-width:1080px; margin:0 auto; padding:32px 20px 80px; }}
 h1 {{ font-size:28px; margin:0 0 4px; }}
 h2 {{ font-size:19px; margin:36px 0 10px; padding-bottom:6px; border-bottom:2px solid #e2e8f0; }}
 .sub {{ color:#4a5568; margin-bottom:24px; }}
 .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; }}
 .kpi {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px; }}
 .kpi .v {{ font-size:23px; font-weight:650; }}
 .kpi .l {{ color:#718096; font-size:13px; }}
 .card {{ background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:12px; margin-top:10px; }}
 img {{ max-width:100%; display:block; }}
 table {{ border-collapse:collapse; width:100%; background:#fff; font-size:13px; }}
 th, td {{ padding:6px 10px; border-bottom:1px solid #edf2f7; text-align:left; }}
 th {{ background:#edf2f7; font-weight:600; position:sticky; top:0; }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .pos {{ color:#1f8a4c; }} .neg {{ color:#c0392b; }}
 .note {{ background:#fffbeb; border:1px solid #f6e05e; padding:10px 14px; border-radius:8px; font-size:13px; }}
 .scroll {{ max-height:460px; overflow:auto; border-radius:10px; border:1px solid #e2e8f0; }}
 p.small {{ font-size:13px; color:#4a5568; }}
</style></head><body><main>
<h1>Обзор рынка акций Мосбиржи</h1>
<div class="sub">Период {date_from} — {date_to} · эталон {index_id} · бумаг в выборке {len(table)}</div>

<div class="kpis">
  <div class="kpi"><div class="v">{signed_pct(idx_ret)}</div><div class="l">{index_id} за период</div></div>
  <div class="kpi"><div class="v">{pct(market_total)}</div><div class="l">средняя доходность (≥60 дней)</div></div>
  <div class="kpi"><div class="v">{above}</div><div class="l">бумаг обогнали {index_id}</div></div>
  <div class="kpi"><div class="v">{num(median_dd, '.1%')}</div><div class="l">медианная макс. просадка</div></div>
  <div class="kpi"><div class="v">{swing:.1%}</div><div class="l">размах {index_id} (от минимума)</div></div>
</div>
<p class="note">{html.escape(fx_note(res['fx']))} Считается только ценовая доходность: дивиденды и налоги не учтены.
Дивидендная доходность по бумагам в отчёте не рассчитывается — в ISS нет готового поля, нужен отдельный
источник дивидендных выплат и защита от пропусков.</p>

<h2>Лидеры роста</h2>
<div class="card"><img src="{images['returns']}" alt="Доходность бумаг"></div>
<p class="small">Средняя доходность бумаги ({pct(market_total)}) — арифметическое по тикерам с историей не короче 60 торговых дней.
Это не рост индекса: индекс взвешен по капитализации, средняя даёт равный вес мелким бумагам.</p>

<h2>Что было бы с 1 ₽</h2>
<div class="card"><img src="{images['normalized']}" alt="Нормированные ряды"></div>

<h2>Риск и доходность</h2>
<div class="card"><img src="{images['riskreturn']}" alt="Риск-доходность"></div>
<p class="small">Волатильность — стандартное отклонение дневных доходностей, приведённое к году (√252).
Максимальная просадка — от локального максимума до минимума. Бета считается к дневным доходностям {index_id}.</p>

<h2>Топ по доходности</h2>
<div class="scroll"><table>{header}<tbody>{table_rows(top)}</tbody></table></div>
<p class="small">Показаны 12 бумаг. Полная таблица — <code>securities_metrics.csv</code> рядом с отчётом.</p>

<h2>Топ по волатильности</h2>
<div class="scroll"><table>{header}<tbody>{table_rows(volatile)}</tbody></table></div>

<h2>Корреляции: что двигается вместе</h2>
<div class="scroll"><table>
<thead><tr><th>Тип</th><th>Бумага A</th><th>Бумага B</th><th>ρ дневных</th></tr></thead>
<tbody>{pairs_html}</tbody></table></div>
<p class="small">Корреляция считается только по дням, когда торговались обе бумаги; меньше 60 общих дней — не считаем.
Низкая корреляция — главный аргумент за диверсификацию.</p>

<h2>Структура индекса</h2>
<div class="card"><img src="{images['weights']}" alt="Веса в индексе"></div>
{weights_note}

<h2>Ликвидность рынка</h2>
<div class="card"><img src="{images['turnover']}" alt="Оборот рынка"></div>
<p class="small">Оборот — сумма по бумагам TQBR в млрд ₽ за торговый день. Дни без торгов не входят.</p>

<h2>Качество данных</h2>
<div class="scroll"><table>
<thead><tr><th>Тикер</th><th>Название</th><th>Первая цена</th><th>Последняя цена</th><th>Торговых дней</th></tr></thead>
<tbody>{short_history}</tbody></table></div>
<p class="small">Мало торговых дней — бумага вышла позже или торгуется нерегулярно. Её доходность считается
по собственному окну и напрямую с остальными не сравнивается.</p>

<h2>Методика и ограничения</h2>
<ul class="small">
<li>Источники: ISS Мосбиржи (режим TQBR, индекс {index_id}) и ЦБ РФ — официальные открытые данные.</li>
<li>Ценовая доходность, без дивидендов и налогов; просадка — по дневным закрытиям.</li>
<li>Ряды склеены по датам торгов: в дни без торгов цена не додумывается.</li>
<li>ETF и паи попадают в выборку вместе с акциями; для акционного анализа их нужно отфильтровать.</li>
<li>Отчёт описывает прошлое, а не прогнозирует доходность.</li>
</ul>
</main></body></html>"""


def main() -> None:
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="HTML-отчёт по данным Мосбиржи")
    parser.add_argument("--db", default=str(here / "data" / "market.duckdb"))
    parser.add_argument("--out", default=str(here / "output"))
    parser.add_argument("--index", default="IMOEX")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db, read_only=True)
    data = load(con)
    res = compute(data, args.index)
    table, idx = res["table"], res["index"]
    idx_ret = float(idx.iloc[-1] / idx.iloc[0] - 1)

    table.round({"total_return": 6, "volatility": 6, "max_drawdown": 6, "beta": 4,
                 "avg_turnover": 2}).to_csv(out_dir / "securities_metrics.csv",
                                            index=False, encoding="utf-8-sig")
    res["pairs"].to_csv(out_dir / "correlation_pairs.csv", index=False, encoding="utf-8-sig")

    images = {
        "returns": chart_returns(table, idx_ret, args.index),
        "normalized": chart_normalized(res, args.index),
        "riskreturn": chart_risk_return(table),
        "weights": chart_weights(res["weights"]) if not res["weights"].empty
                   else _blank("нет данных о составе индекса"),
        "turnover": chart_market_turnover(res),
    }
    date_from, date_to = idx.index[0].date(), idx.index[-1].date()
    (out_dir / "market_report.html").write_text(
        build_html(res, date_from, date_to, args.index, images), encoding="utf-8")
    summary = {
        "index": args.index,
        "period": [str(date_from), str(date_to)],
        "tickers": int(len(table)),
        "index_return": round(idx_ret, 6),
        "report": str(out_dir / "market_report.html"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
