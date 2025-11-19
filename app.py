# app.py
# app.py
import os, time, sys, traceback
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="StockLens Backend", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # zum Testen
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    ticker: str
    as_of: str
    metrics: dict
    verdict: str
    risk_score: int
    ai_summary: str
    disclaimer: str = "Keine Anlageberatung."

def log_exc(e: Exception):
    print("ERROR:", type(e).__name__, str(e), file=sys.stderr)
    import traceback as tb
    tb.print_exc()

@app.get("/health")
def health():
    return {"ok": True}

# -----------------------------------------------------------
# DEBUG_FAKE-Kurzschluss: startet IMMER, auch ohne yfinance
# -----------------------------------------------------------
DEBUG_FAKE = os.getenv("DEBUG_FAKE") == "1"

if DEBUG_FAKE:
    @app.get("/analyze", response_model=AnalyzeResponse)
    def analyze(ticker: str = Query(..., min_length=1, max_length=12)):
        tk = ticker.upper().strip()
        return AnalyzeResponse(
            ticker=tk,
            as_of=time.strftime("%Y-%m-%d"),
            metrics={"pe_ttm": 22.1, "net_margin": 0.27, "price": 100.0, "market_cap": 1e11},
            verdict="fair_to_expensive",
            risk_score=3,
            ai_summary="Fake-Analyse (DEBUG_FAKE=1).",
        )

    @app.get("/probe")
    def probe(ticker: str):
        return {"ticker": ticker.upper(), "metrics": {}}

    @app.get("/diag")
    def diag(ticker: str = "AAPL"):
        return {"ticker": ticker.upper(), "note": "DEBUG_FAKE=1 aktiv"}
# -----------------------------------------------------------
# Nur wenn KEIN Fake aktiv ist, Rest des Backends importieren
# -----------------------------------------------------------
else:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    # HIER kommen alle restlichen Funktionen (make_session, _fi_get,
    # _last_price, _dividend_yield, fetch_metrics, /probe, /diag, /analyze …)

import os, time, sys, traceback, socket
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------
# Requests-Session: Desktop-User-Agent + Retries/Timeouts
# ------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/119.0 Safari/537.36"),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.8,de;q=0.6"
    })
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD", "OPTIONS"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

SESSION = make_session()

# ------------------------------
# FastAPI App + CORS (zum Testen weit offen; später einschränken)
# ------------------------------
app = FastAPI(title="StockLens Backend", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # NUR ZUM DEBUGGEN! Später: exakt deine Framer/Prod-Domains eintragen.
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Models
# ------------------------------
class AnalyzeResponse(BaseModel):
    ticker: str
    as_of: str
    metrics: dict
    verdict: str
    risk_score: int
    ai_summary: str
    disclaimer: str = "Keine Anlageberatung."

# ------------------------------
# Utils
# ------------------------------
def log_exc(e: Exception):
    print("ERROR:", type(e).__name__, str(e), file=sys.stderr)
    traceback.print_exc()

def _num(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def _safe_div(a, b):
    a = _num(a); b = _num(b)
    if a is None or b in (None, 0):
        return None
    return a / b

def _latest(df: pd.DataFrame, key: str):
    """Holt den neuesten verfügbaren Wert für 'key' (robust gegen NaN/Spaltenreihenfolge)."""
    try:
        if df is None or df.empty or key not in df.index:
            return None
        s = df.loc[key]
        s = pd.to_numeric(s, errors="coerce").dropna()
        return _num(s.iloc[0]) if not s.empty else None
    except Exception:
        return None

def _fi_get(fi: dict, *keys):
    """Gibt den ersten vorhandenen Key aus fast_info zurück (unterstützt snake_case & camelCase)."""
    if not isinstance(fi, dict):
        return None
    for k in keys:
        if k in fi:
            return fi.get(k)
    return None


def _fi_get(fi: dict, *keys):
    """Erstes vorhandenes Key-Match aus fast_info zurückgeben (snake_case + camelCase)."""
    if not isinstance(fi, dict):
        return None
    for k in keys:
        if k in fi:
            return fi.get(k)
    return None


# ------------------------------
# yfinance Helfer (immer mit SESSION)
# ------------------------------
def _ticker(ticker: str) -> yf.Ticker:
    # Ticker IMMER mit der Session erzeugen
    return yf.Ticker(ticker, session=SESSION)

def _last_price(t: yf.Ticker):
    """Preis robust: fast_info (snake/camel) -> history(5d) -> history(1mo)."""
    try:
        fi = t.fast_info or {}
        p = _fi_get(
            fi,
            # snake_case Varianten
            "last_price", "last_close", "regular_market_price", "previous_close",
            # camelCase Varianten
            "lastPrice", "lastClose", "regularMarketPrice", "previousClose",
            # weitere sinnvolle Fallbacks
            "open", "dayHigh", "dayLow", "fiftyDayAverage"
        )
        if p is not None:
            return _num(p)
    except Exception:
        pass

    # 2) history 5d
    try:
        h = t.history(period="5d", auto_adjust=False)
        if h is not None and not h.empty:
            v = h["Close"].dropna()
            if not v.empty:
                return _num(v.iloc[-1])
    except Exception:
        pass

    # 3) history 1mo
    try:
        h = t.history(period="1mo", auto_adjust=False)
        if h is not None and not h.empty:
            v = h["Close"].dropna()
            if not v.empty:
                return _num(v.iloc[-1])
    except Exception:
        pass

    return None


def _dividend_yield(t: yf.Ticker, price):
    """Dividendenrendite: fast_info -> Summe der letzten 4 Zahlungen / Preis."""
    try:
        fi = t.fast_info or {}
        # Prüft sowohl snake_case- als auch camelCase-Schlüssel
        y = _fi_get(fi, "dividend_yield", "dividendYield")
        if y is not None:
            return _num(y)
    except Exception:
        pass

    try:
        if price is None:
            return None
        d = t.dividends
        if d is not None and not d.empty:
            # grobe Annäherung: Summe der letzten 4 Dividenden / aktueller Preis
            annual = float(d.tail(4).sum())
            return _safe_div(annual, price)
    except Exception:
        pass

    return None

# ------------------------------
# Daten holen (ohne .info)
# ------------------------------
def fetch_metrics(ticker: str) -> dict:
    """Holt alle relevanten Kennzahlen robust über yfinance."""
    t = _ticker(ticker)  # erzeugt Ticker mit Requests-Session (siehe make_session)

    # === BASISDATEN ===
    price = _last_price(t)

    # Market Cap – prüft snake_case + camelCase
    market_cap = None
    try:
        fi = t.fast_info or {}
        market_cap = _num(_fi_get(fi, "market_cap", "marketCap"))
    except Exception:
        pass

    # === FINANCIAL STATEMENTS (defensiv laden) ===
    try:
        fin = t.financials or pd.DataFrame()      # Income Statement (FY)
    except Exception:
        fin = pd.DataFrame()
    try:
        bs = t.balance_sheet or pd.DataFrame()    # Balance Sheet (FY)
    except Exception:
        bs = pd.DataFrame()

    # === META (Name, Sektor) ===
    name, sector = None, None
    try:
        gi = getattr(t, "get_info", None)
        if callable(gi):
            meta = gi() or {}
            name = meta.get("shortName") or meta.get("longName")
            sector = meta.get("sector")
    except Exception:
        pass

    # === DIVIDENDENRENDITe ===
    dividend_yield = _dividend_yield(t, price)

    # === INCOME STATEMENT KENNZAHLEN ===
    revenue = _latest(fin, "Total Revenue")
    gross_profit = _latest(fin, "Gross Profit")
    operating_income = _latest(fin, "Operating Income")
    net_income = _latest(fin, "Net Income Common Stockholders") or _latest(fin, "Net Income")
    shares_basic = _latest(fin, "Basic Average Shares") or _latest(fin, "BasicAverageShares")

    # Margins
    gross_margin = _safe_div(gross_profit, revenue)
    operating_margin = _safe_div(operating_income, revenue)
    net_margin = _safe_div(net_income, revenue)

    # === BILANZ-KENNZAHLEN ===
    total_debt = _latest(bs, "Total Debt")
    total_equity = _latest(bs, "Total Stockholder Equity") or _latest(bs, "Stockholders Equity")
    debt_to_equity = _safe_div(total_debt, total_equity)

    # === BEWERTUNG ===
    eps = _safe_div(net_income, shares_basic)
    pe_ttm = _safe_div(price, eps)

    # === ERGEBNIS ===
    return {
        "name": name or ticker,
        "sector": sector,
        "market_cap": market_cap,
        "price": price,
        "pe_ttm": pe_ttm,
        "pe_fwd": None,
        "beta": None,  # optional später via SPY-Beta
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "revenue_ttm": revenue,
        "dividend_yield": dividend_yield,
        "debt_to_equity": debt_to_equity,
    }


def risk_from_beta(beta: Optional[float]) -> int:
    if beta is None: return 3
    return 1 if beta < 0.8 else 2 if beta < 1.1 else 3 if beta < 1.4 else 4 if beta < 1.8 else 5

def ai_summary_from_metrics(ticker: str, m: dict) -> str:
    name = m.get("name") or ticker
    verdict = classify_verdict(m)
    parts = []
    parts.append(f"{name} ({ticker}): Kurzfazit.")
    parts.append(f"KGV {m.get('pe_ttm') if m.get('pe_ttm') is not None else 'n/a'}.")
    parts.append(f"netto-Marge {round(m.get('net_margin')*100,1)}%" if m.get('net_margin') is not None else "netto-Marge n/a.")
    parts.append(f"Bewertung wirkt {verdict.replace('_','-')} auf Basis einfacher Multiples.")
    return " ".join(parts)

# ------------------------------
# Endpoints
# ------------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/probe")
def probe(ticker: str):
    """Debug-Endpoint: zeigt rohe, berechnete Metrics (hilft bei yfinance-Aussetzern)."""
    tk = ticker.upper().strip()
    m = fetch_metrics(tk)
    return {"ticker": tk, "metrics": m}

@app.get("/diag")
def diag(ticker: str = "AAPL"):
    """
    Diagnostik: zeigt fast_info (snake+camel), History-Rowcounts und Statement-Rowcounts.
    """
    tk = ticker.upper().strip()
    t = _ticker(tk)
    out = {"ticker": tk}

    # fast_info lesen
    try:
        fi = t.fast_info or {}
        out["fast_info_keys"] = sorted(list(fi.keys()))[:25]

        # kleiner Getter, der snake+camel und Numeric-Cast berücksichtigt
        def getnum(*keys):
            return _num(_fi_get(fi, *keys))

        out["fast_info_sample"] = {
            "price": getnum("last_price", "lastPrice", "regular_market_price", "regularMarketPrice",
                            "previous_close", "previousClose", "open"),
            "market_cap": getnum("market_cap", "marketCap"),
            "dividend_yield": getnum("dividend_yield", "dividendYield"),
            "previous_close": getnum("previous_close", "previousClose"),
            "open": getnum("open", "Open"),
            "dayHigh": getnum("day_high", "dayHigh"),
            "dayLow": getnum("day_low", "dayLow"),
            "fiftyDayAverage": getnum("fifty_day_average", "fiftyDayAverage"),
        }
    except Exception as e:
        out["fast_info_error"] = str(e)

    # history Rowcounts
    try:
        h5 = t.history(period="5d", auto_adjust=False)
        out["history_5d_rows"] = 0 if h5 is None else int(h5.shape[0])
    except Exception as e:
        out["history_5d_error"] = str(e)
    try:
        h1m = t.history(period="1mo", auto_adjust=False)
        out["history_1mo_rows"] = 0 if h1m is None else int(h1m.shape[0])
    except Exception as e:
        out["history_1mo_error"] = str(e)

    # Statements Rowcounts
    try:
        fin = t.financials
        out["financials_rows"] = 0 if fin is None else int(fin.shape[0])
    except Exception as e:
        out["financials_error"] = str(e)
    try:
        bs = t.balance_sheet
        out["balance_sheet_rows"] = 0 if bs is None else int(bs.shape[0])
    except Exception as e:
        out["balance_sheet_error"] = str(e)
    try:
        d = t.dividends
        out["dividends_rows"] = 0 if d is None else int(d.shape[0])
    except Exception as e:
        out["dividends_error"] = str(e)

    return out


@app.get("/analyze", response_model=AnalyzeResponse)
def analyze(ticker: str = Query(..., min_length=1, max_length=12)):
    try:
        tk = ticker.upper().strip()

        # Optionaler Fake-Modus für E2E-Tests ohne yfinance
        if os.getenv("DEBUG_FAKE") == "1":
            return AnalyzeResponse(
                ticker=tk,
                as_of=time.strftime("%Y-%m-%d"),
                metrics={"pe_ttm": 22.1, "net_margin": 0.27, "price": 100.0, "market_cap": 1e11},
                verdict="fair_to_expensive",
                risk_score=3,
                ai_summary="Fake-Analyse (DEBUG_FAKE=1).",
            )

        m = fetch_metrics(tk)

        # Mindestanforderung: wenigstens Preis ODER Market Cap
        if m.get("price") is None and m.get("market_cap") is None:
            return AnalyzeResponse(
                ticker=tk,
                as_of=time.strftime("%Y-%m-%d"),
                metrics=m,
                verdict="fair",
                risk_score=3,
                ai_summary=f"{m.get('name') or tk}: Zu wenige verwertbare Marktdaten verfügbar. "
                           f"Bitte später erneut versuchen oder anderen Ticker testen.",
            )

        verdict = classify_verdict(m)
        risk_score = risk_from_beta(m.get("beta"))
        return AnalyzeResponse(
            ticker=tk,
            as_of=time.strftime("%Y-%m-%d"),
            metrics=m,
            verdict=verdict,
            risk_score=risk_score,
            ai_summary=ai_summary_from_metrics(tk, m),
        )
    except Exception as e:
        log_exc(e)
        raise HTTPException(status_code=500, detail="Interner Fehler bei der Datenverarbeitung.")
