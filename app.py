# app.py
import os, time, sys, traceback
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd, numpy as np, yfinance as yf

app = FastAPI()

# CORS: zum Debug GENAU JETZT weit aufmachen, später auf deine Domains einschränken
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],      # <— NUR ZUM TESTEN
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

def log_exc(e):
    print("ERROR:", type(e).__name__, str(e), file=sys.stderr)
    traceback.print_exc()

@app.get("/health")
def health():
    return {"ok": True}

# ---------- yfinance-robust (ersetzen) ----------

def _num(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return None
        return float(x)
    except Exception:
        return None

def _safe_div(a, b):
    a = _num(a); b = _num(b)
    if a is None or b in (None, 0): return None
    return a / b

def _latest(df: pd.DataFrame, key: str):
    try:
        if df is None or df.empty or key not in df.index: return None
        s = df.loc[key]
        # Nimm den ersten nicht-NaN-Wert (yfinance dreht Spaltenreihenfolge gern mal um)
        s = pd.to_numeric(s, errors="coerce").dropna()
        return _num(s.iloc[0]) if not s.empty else None
    except Exception:
        return None

def _last_price(t: yf.Ticker):
    # 1) fast_info
    try:
        fi = t.fast_info or {}
        p = fi.get("last_price") or fi.get("last_close") or fi.get("regular_market_price")
        if p is not None:
            return _num(p)
    except Exception:
        pass
    # 2) history Fallback
    try:
        h = t.history(period="5d")
        if h is not None and not h.empty:
            return _num(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

def _dividend_yield(t: yf.Ticker, price):
    # fast_info -> ok; sonst aus realen Zahlungen (4Q Summe / Preis)
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
            annual = float(d.tail(4).sum())  # grobe Annäherung
            return _safe_div(annual, price)
    except Exception:
        pass
    return None

def fetch_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)

    # Basis: Preis & Market Cap
    price = _last_price(t)
    market_cap = None
    try:
        fi = t.fast_info or {}
        market_cap = _num(fi.get("market_cap"))
    except Exception:
        pass

    # Statements (defensiv)
    try: fin = t.financials or pd.DataFrame()
    except Exception: fin = pd.DataFrame()
    try: bs  = t.balance_sheet or pd.DataFrame()
    except Exception: bs  = pd.DataFrame()

    # Meta (optional, nie crashen)
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

    # Income Statement
    revenue = _latest(fin, "Total Revenue")
    gross_profit = _latest(fin, "Gross Profit")
    operating_income = _latest(fin, "Operating Income")
    net_income = _latest(fin, "Net Income Common Stockholders") or _latest(fin, "Net Income")
    shares_basic = _latest(fin, "Basic Average Shares") or _latest(fin, "BasicAverageShares")

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
        "beta": None,
        "gross_margin": gross_margin,
        "operating_margin": operating_margin,
        "net_margin": net_margin,
        "revenue_ttm": revenue,          # jährlicher Wert als grobe Näherung
        "dividend_yield": dividend_yield,
        "debt_to_equity": debt_to_equity,
    }

def classify_verdict(m):
    pe = m.get("pe_ttm") or m.get("pe_fwd")
    if pe is None: return "fair"
    if pe < 12: return "cheap"
    if pe < 20: return "fair"
    if pe < 30: return "fair_to_expensive"
    return "expensive"

def risk_from_beta(beta):
    if beta is None: return 3
    return 1 if beta < 0.8 else 2 if beta < 1.1 else 3 if beta < 1.4 else 4 if beta < 1.8 else 5

def ai_summary_from_metrics(ticker, m):
    name = m.get("name") or ticker
    verdict = classify_verdict(m)
    return (f"{name} ({ticker}): Kurzfazit. KGV {m.get('pe_ttm') or 'n/a'}, "
            f"Margin (netto) {m.get('net_margin') or 'n/a'}. "
            f"Bewertung wirkt {verdict.replace('_','-')} auf Basis einfacher Multiples.")

@app.get("/analyze", response_model=AnalyzeResponse)
def analyze(ticker: str = Query(..., min_length=1, max_length=10)):
    try:
        tk = ticker.upper().strip()
       m = fetch_metrics(tk)
# Mindestanforderung: wir brauchen wenigstens einen Preis ODER Market Cap
if m.get("price") is None and m.get("market_cap") is None:
    # gib wenigstens einen schlanken Dummy zurück – Frontend bleibt funktionsfähig
    return AnalyzeResponse(
        ticker=tk,
        as_of=time.strftime("%Y-%m-%d"),
        metrics=m,
        verdict="fair",
        risk_score=3,
        ai_summary=f"{m.get('name') or tk}: Zu wenige verwertbare Marktdaten verfügbar. Bitte später erneut versuchen oder anderen Ticker testen.",
    )
        verdict = classify_verdict(m)
        risk_score = risk_from_beta(m.get("beta"))
        return AnalyzeResponse(
            ticker=tk,
            as_of=time.strftime("%Y-%m-%d"),
            metrics=m,
            verdict=verdict,
            risk_score=risk_score,
            ai_summary=ai_summary_from_metrics(tk, m)
        )
    except HTTPException:
        raise
    except Exception as e:
        log_exc(e)
        raise HTTPException(status_code=500, detail="Interner Fehler bei der Datenverarbeitung.")
