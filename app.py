import os, time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf

# ---- Konfig ----
ALLOWED_ORIGINS = [
    "https://stocklens-backend.onrender.com",  # ersetze: deine Framer-URL/Custom-Domain
    "https://lower-project-897650.framer.app/actual"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

CACHE = {}           # {ticker: (ts, payload)}
TTL_SECONDS = 6*3600 # 6h

class AnalyzeResponse(BaseModel):
    ticker: str
    as_of: str
    metrics: dict
    verdict: str
    risk_score: int
    ai_summary: str
    disclaimer: str = "Keine Anlageberatung."

def fetch_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = getattr(t, "info", {}) or {}
    return {
        "name": info.get("shortName"),
        "sector": info.get("sector"),
        "market_cap": info.get("marketCap"),
        "pe_ttm": info.get("trailingPE"),
        "pe_fwd": info.get("forwardPE"),
        "beta": info.get("beta"),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin": info.get("profitMargins"),
        "revenue_ttm": info.get("totalRevenue"),
        "dividend_yield": info.get("dividendYield"),
        "debt_to_equity": info.get("debtToEquity")
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

# --- Platzhalter: Hier könntest du OpenAI callen (weggelassen für Kürze) ---
def ai_summary_from_metrics(ticker, m):
    name = m.get("name") or ticker
    verdict = classify_verdict(m)
    return (
        f"{name} ({ticker}): Kurzfazit. Sektor {m.get('sector') or 'n/a'}, "
        f"KGV {m.get('pe_ttm') or m.get('pe_fwd') or 'n/a'}, Beta {m.get('beta') or 'n/a'}. "
        f"Bewertung wirkt {verdict.replace('_','-')} auf Basis einfacher Multiples. "
        "Prüfpunkte: Wachstum/Nachhaltigkeit der Margen, Verschuldung."
    )

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/analyze", response_model=AnalyzeResponse)
def analyze(ticker: str = Query(..., min_length=1, max_length=10)):
    tk = ticker.upper().strip()
    now = time.time()
    if tk in CACHE and now - CACHE[tk][0] < TTL_SECONDS:
        return CACHE[tk][1]

    try:
        metrics = fetch_metrics(tk)
        if not any(metrics.values()):
            raise HTTPException(status_code=404, detail="Ticker nicht gefunden oder Daten nicht verfügbar.")
        verdict = classify_verdict(metrics)
        risk_score = risk_from_beta(metrics.get("beta"))
        payload = AnalyzeResponse(
            ticker=tk,
            as_of=time.strftime("%Y-%m-%d"),
            metrics=metrics,
            verdict=verdict,
            risk_score=risk_score,
            ai_summary=ai_summary_from_metrics(tk, metrics)
        ).model_dump()
        CACHE[tk] = (now, payload)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interner Fehler: {str(e)}")
