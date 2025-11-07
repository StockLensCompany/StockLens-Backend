# app.py
import os, time, sys, traceback
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------------------------------
# FastAPI App + CORS (zum Testen weit offen; später Domains einschränken)
# ------------------------------
app = FastAPI(title="StockLens Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # NUR ZUM DEBUGGEN! Später: exakt deine Framer/Prod-Domains eintragen.
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
    """Holt den neuesten verfügbaren Wert für 'key' aus einem yfinance DataFrame (robust gegen NaN/Spaltenreihenfolge)."""
    try:
        if df is None or df.empty or key not in df.index:
            return None
        s = df.loc[key]
        s = pd.to_numeric(s, errors="coerce").dropna()
        return _num(s.iloc[0]) if not s.empty else None
    except Exception:
        return None

def _last_price(t: yf.Ticker):
    """Preis robust bestimmen: fast_info -> history(5d) -> history(1mo)."""
    # 1) fast_info
    try:
        fi = t.fast_info or {}
        p = fi.get("last_price") or fi.get("last_close") or fi.get("regular_market_price")
        if p is not None:
            return _num(p)
    except Exception:
        pass
    # 2) history 5d
    try:
        h = t.history(period="5d")
        if h is not None and not h.empty:
            v = h["Close"].dropna()
            if not v.empty:
                return _num(v.iloc[-1])
    except Exception:
        pass
    # 3) history 1mo
    try:
        h = t.history(period="1mo")
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
        y = fi.get("dividend_yield")
        if y is not None:
            return _num(y)
    except Exception:
        pass
    try:
        if price is None:
            return None
        d = t.dividends
        if d is not None and not d.empty:
            annual = float(d.tail(4).sum())  # grobe Annäherung aus den letzten 4 Zahlungen
            return _safe_div(annual, price)
    except Exception:
        pass
    return None

# ------------------------------
# Daten holen (ohne .info)
# ------------------------------
def fetch_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)

    # Basis
    price = _last_price(t)
    market_cap = None
    try:
        fi = t.fast_info or {}
        market_cap = _num(fi.get("market_cap"))
    except Exception:
        pass

    # Statements defensiv lesen
    try: fin = t.financials or pd.DataFrame()      # Income Statement (FY)
    except Exception: fin = pd.DataFrame()
    try: bs  = t.balance_sheet or pd.DataFrame()   # Balance Sheet (FY)
    except Exception: bs  = pd.DataFrame()

    # Meta (optional)
    name, sector = None, None
    try:
        gi = getattr(t, "get_info", None)
        if callable(gi):
            meta = gi() or {}
            name = meta.get("shortName") or meta.get("longName")
            sector = meta.get("sector")
    except Exception:
        pass

    # Dividendenrendite
    dividend_yield = _dividend_yield(t, price)

    # Income Statement Kennzahlen
    revenue = _latest(fin, "Total Revenue")
    gross_profit = _latest(fin, "Gross Profit")
    operating_income = _latest(fin, "Operating Income")
    net_income = _latest(fin, "Net Income Common Stockholders") or _latest(fin, "Net Income")
    shares_basic = _latest(fin, "Basic Average Shares") or _latest(fin, "BasicAverageShares")

    # Margins
    gross_margin = _safe_div(gross_profit, revenue)
    operating_margin = _safe_div(operating_income, revenue)
    net_margin = _safe_div(net_income, revenue)

    # Bilanz
    total_debt = _latest(bs, "Total Debt")
    total_equity = _latest(bs, "Total Stockholder Equity") or _latest(bs, "Stockholders Equity")
    debt_to_equity = _safe_div(total_debt, total_equity)

    # P/E (einfach)
    eps = _safe_div(net_income, shares_basic)
    pe_ttm = _safe_div(price, eps)

    return {
        "name": name or ticker,
        "sector": sector,
        "market_cap": market_cap,
        "price": price,
        "pe_ttm": pe_ttm,
        "pe_fwd": None,
        "beta": None,  # Optional später via History vs SPY approximieren
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "revenue_ttm": revenue,          # FY-Wert als grobe Näherung
        "dividend_yield": dividend_yield,
        "debt_to_equity": debt_to_equity,
    }

# ------------------------------
# Bewertung + Risiko
# ------------------------------
def classify_verdict(m: dict) -> str:
    pe = m.get("pe_ttm") or m.get("pe_fwd")
    if pe is None: return "fair"
    if pe < 12: return "cheap"
    if pe < 20: return "fair"
    if pe < 30: return "fair_to_expensive"
    return "expensive"

def risk_from_beta(beta: Optional[float]) -> int:
    if beta is None: return 3
    return 1 if beta < 0.8 else 2 if beta < 1.1 else 3 if beta < 1.4 else 4 if beta < 1.8 else 5

def ai_summary_from_metrics(ticker: str, m: dict) -> str:
    name = m.get("name") or ticker
    verdict = classify_verdict(m)
    return (
        f"{name} ({ticker}): Kurzfazit. "
        f"KGV {m.get('pe_ttm') if m.get('pe_ttm') is not None else 'n/a'}, "
        f"netto-Marge {round(m.get('net_margin')*100,1)}% " if m.get('net_margin') is not None else "netto-Marge n/a. "
        f"Bewertung wirkt {verdict.replace('_','-')} auf Basis einfacher Multiples."
    )

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

        # Mindestanforderung: wenigstens Preis ODER Market Cap, sonst kompakte Hinweis-Antwort (kein 502!)
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
        # Keine Interna an den Client leaken:
        raise HTTPException(status_code=500, detail="Interner Fehler bei der Datenverarbeitung.")
