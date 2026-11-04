"""
每天由 GitHub Actions 執行：抓 11 個板塊 ETF 與 SPY 的週線，
用跟 Streamlit 版相同的 RRG 近似算法算出相對強度與相對動能，
存成 data/rrg.json，給 usbbtrading.com/rrg/ 的網頁讀取。
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

BENCH = "SPY"
SECTORS = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLV", "XLE", "XLB", "XLP", "XLU", "XLRE"]
WINDOW = 14
MOM_WINDOW = 6
OUT = Path(__file__).resolve().parent.parent / "data" / "rrg.json"


def download(tickers):
    raw = yf.download(tickers, period="5y", interval="1wk", auto_adjust=True,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(close, pd.DataFrame):
        close = close.to_frame()
    if len(tickers) == 1:
        close.columns = tickers
    return close


def load_prices():
    tickers = SECTORS + [BENCH]
    close = download(tickers)
    for attempt in range(4):
        missing = [t for t in tickers if t not in close or close[t].dropna().empty]
        if not missing:
            break
        time.sleep(5 * (attempt + 1))
        for t in missing:
            try:
                one = download([t])
                if not one.empty and one[t].notna().any():
                    close[t] = one[t]
            except Exception as e:
                print("retry failed", t, e)
    close = close.dropna(how="all").ffill()
    good = [t for t in tickers if t in close and close[t].notna().any()]
    if BENCH not in good or len(good) - 1 < len(SECTORS):
        raise RuntimeError(f"price download incomplete: {good}")
    return close[good]


def compute(close):
    bench = close[BENCH]
    ratio, mom = {}, {}
    for t in SECTORS:
        rs = 100 * close[t] / bench
        r = 100 * rs.ewm(span=WINDOW).mean() / rs.rolling(WINDOW * 2).mean()
        m = 100 * r / r.rolling(MOM_WINDOW).mean()
        ratio[t], mom[t] = r, m
    ratio, mom = pd.DataFrame(ratio), pd.DataFrame(mom)
    ok = ratio.notna().all(axis=1) & mom.notna().all(axis=1)
    return ratio[ok], mom[ok]


def main():
    close = load_prices()
    ratio, mom = compute(close)
    if len(ratio) < 60:
        raise RuntimeError(f"too few rows: {len(ratio)}")
    data = {
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "bench": BENCH,
        "dates": [d.strftime("%Y-%m-%d") for d in ratio.index],
        "ratio": {t: [round(float(v), 3) for v in ratio[t]] for t in SECTORS},
        "mom": {t: [round(float(v), 3) for v in mom[t]] for t in SECTORS},
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} rows={len(ratio)} last={data['dates'][-1]}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 抓不到資料就失敗退出，保留上一份 rrg.json，網頁照舊能用
        print("ERROR:", e)
        sys.exit(1)
