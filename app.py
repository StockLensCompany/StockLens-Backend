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

# ---------- yfinance-robust ----------
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
        if df is None or df.empty: return None
        if key not in df.index: return None
        s = df.loc[key]
        return _num(s.iloc[0]) if hasattr(s, "iloc") else _num(s)
    except Exception:
        return None

def fetch_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)

    # fast_info
    try:
        fi = t.fast_info or {}
    except Exception:
        fi = {}

    # statements defensiv
    try: fin = t.financials or pd.DataFrame()
    except Exception: fin = pd.DataFrame()
    try: bs  = t.balance_sheet or pd.DataFrame()
    except Exception: bs  = pd.DataFrame()

    # meta optional
    name, sector = None, None
    try:
        gi = getattr(t, "get_info", None)
        if callable(gi):
            meta = gi() or {}
            name = meta.get("shortName") or meta.get("longName")
            sector = meta.get("sector")
    except Exception:
        pass

    price = _num(fi.get("last_price"))
    market_cap = _num(fi.get("market_cap"))
    div_val = _num(fi.get("last_dividend_value"))
    dividend_yield = _safe_div(div_val, price)

    revenue = _latest(fin, "Total Revenue")
    gross_profit = _latest(fin, "Gross Profit")
    operating_income = _latest(fin, "Operating Income")
    net_income = _latest(fin, "Net Income Common Stockholders") or _latest(fin, "Net Income")
    shares_basic = _latest(fin, "Basic Average Shares") or _latest(fin, "BasicAverageShares")

    gross_margin = _safe_div(gross_profit, revenue)
    operating_margin = _safe_div(operating_income, revenue)
    net_margin = _safe_div(net_income, revenue)

    total_debt = _latest(bs, "Total Debt")
    total_equity = _latest(bs, "Total Stockholder Equity") or _latest(bs, "Stockholders Equity")
    debt_to_equity = _safe_div(total_debt, total_equity)

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
        "revenue_ttm": revenue,
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
        have_any = any(v is not None for k,v in m.items() if k not in ("name","sector"))
        if not have_any:
            raise HTTPException(status_code=502, detail="Upstream-Datenquelle lieferte keine verwertbaren Werte.")
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
