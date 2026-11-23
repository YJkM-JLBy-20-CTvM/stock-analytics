"""Официальный курс USD ЦБ РФ. Файл отдаётся в cp1251."""
from __future__ import annotations

import ssl
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

URL = "https://www.cbr.ru/scripts/XML_daily.asp?date_req={:%d/%m/%Y}"
CA_BUNDLE = Path(__file__).with_name("certs") / "russian_trusted_root_ca.pem"


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    if CA_BUNDLE.exists():
        context.load_verify_locations(cafile=str(CA_BUNDLE))
    return context


def fetch_usd_rate(on: date) -> float | None:
    req = urllib.request.Request(URL.format(on), headers={"User-Agent": "moex-market-report/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
        root = ET.fromstring(resp.read())
    for valute in root.findall("Valute"):
        if valute.get("CharCode") == "USD" or valute.findtext("CharCode") == "USD":
            nominal = float(valute.findtext("Nominal").replace(",", "."))
            value = float(valute.findtext("Value").replace(",", "."))
            return value / nominal
    return None
