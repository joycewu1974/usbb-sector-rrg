"""
美股板塊輪動 RRG（動畫版）— Prototype v2
=======================================

跟前一版的差異：
    1. 真正的動畫播放：右下角有播放鍵＋時間滑桿，可以看板塊泡泡
       隨時間在四象限之間移動的完整過程，不再只是靜態尾跡線。
    2. 泡泡大小 = 資金流量（成交金額 price × volume 的滾動平均），
       美股沒有台灣「三大法人買賣超」那種公開籌碼資料，但 Volume
       本身就是市場真實資金進出的紀錄，用成交金額當「大戶／資金
       關注度」的代理指標是業界常見做法（愈多錢流入，泡泡愈大）。

X軸 = RS-Ratio（相對大盤強度，100為中性）
Y軸 = RS-Momentum（強度變化動能，100為中性）
四象限：領先 Leading／轉弱 Weakening／落後 Lagging／改善 Improving

★ 目前為模擬資料（含模擬成交量），週數只是序號(第0週...第N週)，
   不對應真實日期。接上 yfinance 之後，滑桿會自動顯示真實的
   每週日期（因為 week_idx 只是索引，改用真實資料的 DataFrame
   index 即可，畫圖邏輯完全不用改）。

★ 接真實資料只要把 fetch_price_volume_data() 換成 yfinance 版本即可，
   下面所有計算、動畫、畫圖邏輯完全不用改。

真實資料範例（本機執行，需要網路）：
    import yfinance as yf
    def fetch_price_volume_data(tickers, period="2y", interval="1wk"):
        data = yf.download(tickers, period=period, interval=interval, group_by="ticker")
        out = {}
        for t in tickers:
            out[t] = pd.DataFrame({
                "close": data[t]["Close"],
                "volume": data[t]["Volume"],
            }).dropna()
        return out

執行方式：
    pip3 install yfinance streamlit plotly pandas numpy
    python3 -m streamlit run us_sector_rotation_app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

st.set_page_config(page_title="美股板塊輪動 RRG（動畫版）", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1F2B; }
    </style>
    """,
    unsafe_allow_html=True,
)

BENCHMARK = "SPY"
SECTOR_ETFS = {
    "XLK": "科技", "XLF": "金融", "XLE": "能源", "XLV": "醫療",
    "XLI": "工業", "XLY": "非必需消費", "XLP": "必需消費",
    "XLB": "原物料", "XLU": "公用事業", "XLRE": "房地產", "XLC": "通訊服務",
}

WEEKS = 65           # 模擬資料週數（要多留一些給rolling window吃掉）
ANIM_FRAMES = 52     # 預設動畫frame數（取最近52週），可被下方時間區間選單覆蓋

RANGE_OPTIONS = {"1個月": 4, "3個月": 13, "6個月": 26, "1年": 52}

# 每個板塊的代表性重倉股（手動整理，非即時抓取的官方持股清單）。
# 這份清單會過時，建議每季手動檢查一次是否需要更新；
# 之後若要更精準，可以換成付費資料源即時拉取官方持股。
SECTOR_CONSTITUENTS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "BLK"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC"],
    "XLV": ["UNH", "LLY", "JNJ", "ABBV", "MRK", "TMO"],
    "XLI": ["CAT", "HON", "UNP", "RTX", "GE", "BA"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
    "XLP": ["PG", "KO", "PEP", "WMT", "COST", "PM"],
    "XLB": ["LIN", "SHW", "FCX", "NEM", "APD", "ECL"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
    "XLRE": ["PLD", "AMT", "EQIX", "PSA", "O", "SPG"],
    "XLC": ["META", "GOOGL", "NFLX", "DIS", "CMCSA", "TMUS"],
}


# ----------------------------------------------------------------------
# 1. 資料層：真實資料版（yfinance），失敗時自動退回模擬資料
# ----------------------------------------------------------------------
def _mock_price_volume_data(tickers, weeks=WEEKS, seed=7):
    """模擬資料，當yfinance抓不到資料時的備援，避免頁面直接壞掉。"""
    rng = np.random.default_rng(seed)
    out = {}

    spy_close = 400 * np.exp(np.cumsum(rng.normal(0.002, 0.015, weeks)))
    out[BENCHMARK] = pd.DataFrame({
        "close": spy_close,
        "volume": rng.normal(5e7, 5e6, weeks).clip(min=1e7),
    })

    for t in tickers:
        drift = rng.normal(0.0, 0.0025)
        cycle_len = rng.integers(12, 24)
        phase = rng.uniform(0, 2 * np.pi)
        noise = rng.normal(0, 0.018, weeks)
        cycle = 0.012 * np.sin(np.linspace(0, weeks / cycle_len * 2 * np.pi, weeks) + phase)
        rel_drift = drift + cycle + noise
        close = spy_close[0] * 0.6 * np.exp(np.cumsum(rel_drift)) * (spy_close / spy_close[0]) ** 0.3

        base_vol = rng.normal(1.5e7, 3e6, weeks).clip(min=1e6)
        burst_weeks = rng.choice(weeks, size=rng.integers(2, 5), replace=False)
        for bw in burst_weeks:
            span = slice(max(0, bw - 1), min(weeks, bw + 2))
            base_vol[span] *= rng.uniform(1.8, 3.2)

        out[t] = pd.DataFrame({"close": close, "volume": base_vol})
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _download_yfinance_data(all_tickers: tuple, period: str = "2y", interval: str = "1wk"):
    """實際打yfinance抓資料，快取1小時避免重複下載。"""
    data = yf.download(list(all_tickers), period=period, interval=interval,
                        group_by="ticker", progress=False, threads=True)
    out = {}
    for t in all_tickers:
        try:
            df = pd.DataFrame({
                "close": data[t]["Close"],
                "volume": data[t]["Volume"],
            }).dropna()
        except (KeyError, TypeError):
            df = pd.DataFrame(columns=["close", "volume"])
        out[t] = df
    return out


def fetch_price_volume_data(tickers, weeks=WEEKS, seed=7):
    """
    回傳 (dict[ticker] -> pd.DataFrame(columns=["close", "volume"]), is_mock: bool)。
    優先使用yfinance真實股價；抓不到資料(離線/套件沒裝/API限流)時
    自動退回模擬資料，並在頁面上顯示提示，不會讓App直接壞掉。
    """
    if not YFINANCE_AVAILABLE:
        st.warning("尚未安裝 yfinance，目前顯示的是模擬資料。請在終端機執行："
                    "`pip3 install yfinance` 後重新整理頁面。")
        return _mock_price_volume_data(tickers, weeks, seed), True

    all_tickers = tuple([BENCHMARK] + list(tickers))
    try:
        out = _download_yfinance_data(all_tickers)
    except Exception as e:
        st.warning(f"抓取yfinance資料時發生錯誤，暫時顯示模擬資料。錯誤訊息：{e}")
        return _mock_price_volume_data(tickers, weeks, seed), True

    # 檢查資料是否足夠（至少要有benchmark跟每檔ETF的資料）
    if out.get(BENCHMARK, pd.DataFrame()).empty or all(out[t].empty for t in tickers):
        st.warning("yfinance回傳的資料是空的（可能是網路問題或暫時被限流），"
                    "目前顯示模擬資料，稍後可重新整理頁面再試一次。")
        return _mock_price_volume_data(tickers, weeks, seed), True

    return out, False


@st.cache_data(ttl=3600, show_spinner=False)
def _download_yfinance_daily(all_tickers: tuple, period: str = "15mo"):
    """Trend Template要用日線(50/150/200日均線、52週高低點)，跟RRG的週線分開快取抓取。"""
    data = yf.download(list(all_tickers), period=period, interval="1d",
                        group_by="ticker", progress=False, threads=True)
    out = {}
    for t in all_tickers:
        try:
            df = pd.DataFrame({
                "close": data[t]["Close"],
                "volume": data[t]["Volume"],
            }).dropna()
        except (KeyError, TypeError):
            df = pd.DataFrame(columns=["close", "volume"])
        out[t] = df
    return out


def fetch_daily_data(tickers, weeks_as_days=380, seed=11):
    """
    回傳 (dict[ticker] -> pd.DataFrame(close, volume) 日線, is_mock: bool)。
    邏輯跟 fetch_price_volume_data 一樣：優先真實資料，抓不到就退回模擬資料。
    """
    if not YFINANCE_AVAILABLE:
        return _mock_price_volume_data(tickers, weeks_as_days, seed), True

    all_tickers = tuple([BENCHMARK] + list(tickers))
    try:
        out = _download_yfinance_daily(all_tickers)
    except Exception:
        return _mock_price_volume_data(tickers, weeks_as_days, seed), True

    if out.get(BENCHMARK, pd.DataFrame()).empty or all(out[t].empty for t in tickers):
        return _mock_price_volume_data(tickers, weeks_as_days, seed), True

    return out, False


# ----------------------------------------------------------------------
# 2. RS-Ratio / RS-Momentum / 資金流量大小 計算
# ----------------------------------------------------------------------
def compute_rrg_with_flow(prices: dict, benchmark: str, window: int = 10,
                            mom_period: int = 4, flow_window: int = 6):
    bench_close = prices[benchmark]["close"]
    records = {}
    for ticker, df in prices.items():
        if ticker == benchmark:
            continue
        close = df["close"]
        volume = df["volume"]

        rs = close / bench_close
        rs_z = (rs - rs.rolling(window).mean()) / rs.rolling(window).std()
        rs_ratio = 100 + rs_z * 2

        rs_ratio_chg = rs_ratio.diff(mom_period)
        mom_z = (rs_ratio_chg - rs_ratio_chg.rolling(window).mean()) / rs_ratio_chg.rolling(window).std()
        rs_momentum = 100 + mom_z * 2

        dollar_flow = (close * volume).rolling(flow_window).mean()

        rec = pd.DataFrame({
            "rs_ratio": rs_ratio,
            "rs_momentum": rs_momentum,
            "dollar_flow": dollar_flow,
        }).dropna()
        records[ticker] = rec
    return records


def label_quadrant(x, y):
    if x >= 100 and y >= 100:
        return "領先 Leading"
    if x >= 100 and y < 100:
        return "轉弱 Weakening"
    if x < 100 and y < 100:
        return "落後 Lagging"
    return "改善 Improving"


QUADRANT_COLORS = {
    "領先 Leading": "#3FA796",
    "轉弱 Weakening": "#D4AF37",
    "落後 Lagging": "#C0435A",
    "改善 Improving": "#5B8DBE",
}

NAVY_BG = "#132A3A"
NAVY_PANEL = "#0E1F2B"
GOLD = "#D4AF37"
GRID_LINE = "rgba(212,175,55,0.15)"


# ----------------------------------------------------------------------
# 2b. Trend Template（8條件）與大盤健康度
# ----------------------------------------------------------------------
def compute_trend_template(close: pd.Series) -> dict:
    """
    對照Joyce的黑帶Trend Template skill，用日線收盤價算出條件1-7
    （條件8的RS另外用compute_rs_percentile算，因為要跟同組股票互相比較）。
    資料不足200個交易日時回傳 None，避免算出不可靠的均線。
    """
    if len(close) < 210:
        return None

    price = close.iloc[-1]
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()

    sma50_last = sma50.iloc[-1]
    sma150_last = sma150.iloc[-1]
    sma200_last = sma200.iloc[-1]
    sma200_1m_ago = sma200.iloc[-22] if len(sma200.dropna()) > 22 else np.nan

    high_52w = close.tail(252).max()
    low_52w = close.tail(252).min()
    pct_above_low = (price / low_52w - 1) * 100
    pct_below_high = (1 - price / high_52w) * 100

    cond = {
        "c1_price_above_150_200": bool(price > sma150_last and price > sma200_last),
        "c2_150_above_200": bool(sma150_last > sma200_last),
        "c3_200_rising": bool(not np.isnan(sma200_1m_ago) and sma200_last > sma200_1m_ago),
        "c4_50_above_150_200": bool(sma50_last > sma150_last and sma50_last > sma200_last),
        "c5_price_above_50": bool(price > sma50_last),
        "c6_30pct_above_52w_low": bool(pct_above_low >= 30),
        "c7_within_25pct_of_52w_high": bool(pct_below_high <= 25),
        "price": price,
        "pct_above_52w_low": pct_above_low,
        "pct_below_52w_high": pct_below_high,
    }
    cond["ma_structure_pass"] = all(cond[k] for k in
                                     ["c1_price_above_150_200", "c2_150_above_200",
                                      "c3_200_rising", "c4_50_above_150_200", "c5_price_above_50"])
    cond["price_position_pass"] = cond["c6_30pct_above_52w_low"] and cond["c7_within_25pct_of_52w_high"]
    return cond


def compute_rs_percentile(closes: dict, spy_close: pd.Series, lookback: int = 252) -> dict:
    """
    近似版RS：算每檔股票過去lookback個交易日相對SPY的報酬，
    再把這組股票互相排名轉成0-100百分位。
    ⚠️ 這不是IBD那種全市場RS Rating，只是同一個板塊成分股之間的相對排名，僅供參考。
    """
    rel_returns = {}
    spy_ret = spy_close.iloc[-1] / spy_close.iloc[-min(lookback, len(spy_close))] - 1
    for ticker, close in closes.items():
        if len(close) < min(lookback, 50):
            continue
        stock_ret = close.iloc[-1] / close.iloc[-min(lookback, len(close))] - 1
        rel_returns[ticker] = stock_ret - spy_ret

    if not rel_returns:
        return {}
    ranked = sorted(rel_returns.items(), key=lambda kv: kv[1])
    n = len(ranked)
    return {ticker: round((i + 1) / n * 100) for i, (ticker, _) in enumerate(ranked)}


def compute_market_health(spy_close: pd.Series) -> dict:
    """大盤（SPY）自己的健康度速覽：站不站得上50/200日均線、200日均線有沒有上揚。"""
    if len(spy_close) < 210:
        return {"label": "資料不足，暫無法判斷大盤健康度", "ok": None}

    price = spy_close.iloc[-1]
    sma50 = spy_close.rolling(50).mean().iloc[-1]
    sma200_series = spy_close.rolling(200).mean()
    sma200 = sma200_series.iloc[-1]
    sma200_1m_ago = sma200_series.iloc[-22] if len(sma200_series.dropna()) > 22 else np.nan

    above_50 = price > sma50
    above_200 = price > sma200
    rising_200 = not np.isnan(sma200_1m_ago) and sma200 > sma200_1m_ago

    score = sum([above_50, above_200, rising_200])
    if score == 3:
        label = "大盤健康度：偏多頭。SPY站上50日與200日均線，200日均線持續上揚，適合積極尋找輪動機會。"
        ok = True
    elif score == 0:
        label = "大盤健康度：偏空頭。SPY跌破50日與200日均線，200日均線走平或向下，建議以防守為主、放慢進場節奏。"
        ok = False
    else:
        label = "大盤健康度：中性/轉換中。均線訊號不一致，建議個股輪動訊號要更嚴謹確認，不宜全面積極進攻。"
        ok = None
    return {"label": label, "ok": ok, "above_50": bool(above_50),
            "above_200": bool(above_200), "rising_200": bool(rising_200)}


# ----------------------------------------------------------------------
# 3. 整理成動畫用的長格式資料
# ----------------------------------------------------------------------
def build_animation_frame_data(rrg_data: dict, name_map: dict, n_frames: int = ANIM_FRAMES,
                                highlight_tickers: set = frozenset()):
    rows = []
    # 取所有ticker共同的最後 n_frames 個時間點
    min_len = min(len(df) for df in rrg_data.values())
    n_frames = min(n_frames, min_len)

    week_labels = None
    for ticker, df in rrg_data.items():
        tail = df.tail(n_frames)
        # 有真實日期索引(yfinance)就記錄下來，之後滑桿顯示真實週別而非序號
        if week_labels is None:
            try:
                week_labels = [d.strftime("%Y-%m-%d") for d in tail.index]
            except AttributeError:
                week_labels = [str(i) for i in range(len(tail))]
        tail = tail.reset_index(drop=True)
        label = f"{ticker} {name_map[ticker]}" if ticker in name_map else ticker
        symbol = "star" if ticker in highlight_tickers else "circle"
        for i, row in tail.iterrows():
            rows.append({
                "week_idx": i,
                "ticker": ticker,
                "label": label,
                "rs_ratio": row.rs_ratio,
                "rs_momentum": row.rs_momentum,
                "dollar_flow": row.dollar_flow,
                "quadrant": label_quadrant(row.rs_ratio, row.rs_momentum),
                "symbol": symbol,
            })
    long_df = pd.DataFrame(rows)

    # 每個frame內部把資金流量正規化成泡泡大小(15~55)，才會有「大戶越多越大」的視覺對比
    # 用 groupby().transform() 而不是 groupby().apply()，避免不同pandas版本在
    # apply回傳DataFrame時把分組欄位處理成index，導致後面用不到 week_idx 欄位。
    def _size_from_flow(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-9:
            return pd.Series(30.0, index=s.index)
        return 15 + 40 * (s - lo) / (hi - lo)

    long_df["size"] = long_df.groupby("week_idx")["dollar_flow"].transform(_size_from_flow)
    return long_df, n_frames, week_labels


def compute_quadrant_streak(long_df: pd.DataFrame) -> dict:
    """
    回傳 {ticker: 已在目前象限連續停留的週數}。
    從最新一週往回數，只要象限跟最新一週相同就繼續累加，
    一遇到不同象限就停止——用來判斷「剛換象限」還是「已經穩定一段時間」。
    """
    streaks = {}
    for ticker, g in long_df.groupby("ticker"):
        g = g.sort_values("week_idx")
        quadrants = g["quadrant"].tolist()
        current = quadrants[-1]
        streak = 0
        for q in reversed(quadrants):
            if q == current:
                streak += 1
            else:
                break
        streaks[ticker] = streak
    return streaks


def compute_transition_alerts(long_df: pd.DataFrame, lookback: int = 3) -> dict:
    """
    回傳 {ticker: {"from": 目前象限, "to": 推估下一步象限}}，
    只包含「照最近lookback週的移動速度推算一步之後，會跨到不同象限」的標的——
    用來提醒「快要換象限了」，不用等它真的跨過去才發現。
    這是根據近期趨勢的簡單線性推估，不是預測，僅供提醒留意。
    """
    alerts = {}
    for ticker, g in long_df.groupby("ticker"):
        g = g.sort_values("week_idx")
        if len(g) < lookback + 1:
            continue
        recent = g.tail(lookback + 1)
        x0, x1 = recent.iloc[0]["rs_ratio"], recent.iloc[-1]["rs_ratio"]
        y0, y1 = recent.iloc[0]["rs_momentum"], recent.iloc[-1]["rs_momentum"]
        vx = (x1 - x0) / lookback
        vy = (y1 - y0) / lookback
        current_quadrant = recent.iloc[-1]["quadrant"]
        projected_quadrant = label_quadrant(x1 + vx, y1 + vy)
        if projected_quadrant != current_quadrant:
            alerts[ticker] = {"from": current_quadrant, "to": projected_quadrant}
    return alerts


# ----------------------------------------------------------------------
# 4. 動畫圖表
#    重點：每支ETF固定用「同一條trace」貫穿所有frame（只更新座標/大小/顏色），
#    這樣Plotly才能在frame之間做平滑補間(tween)，而不是每次重繪造成的跳動。
#    象限只用來決定顏色，不能再拿來分組trace，否則每個frame的trace筆數
#    會變動，補間就會失效、動畫就會頓。
# ----------------------------------------------------------------------
def build_animated_figure(long_df: pd.DataFrame, n_frames: int, ticker_order: list,
                           week_labels: list = None):
    if not week_labels or len(week_labels) != n_frames:
        week_labels = [str(w) for w in range(n_frames)]
    x_min, x_max = long_df.rs_ratio.min() - 1.5, long_df.rs_ratio.max() + 1.5
    y_min, y_max = long_df.rs_momentum.min() - 1.5, long_df.rs_momentum.max() + 1.5

    def make_ticker_traces(week_idx):
        """
        每支ETF產生一對 [光暈trace, 主體trace]，順序固定，
        這樣兩條trace才能跟frame資料一一對應、做平滑補間。
        """
        d = long_df[long_df["week_idx"] == week_idx].set_index("ticker")
        traces = []
        for ticker in ticker_order:
            row = d.loc[ticker]
            color = QUADRANT_COLORS[row.quadrant]
            size = row["size"]
            symbol = row["symbol"]
            is_custom = symbol == "star"
            # 外層光暈：同座標、更大、更透明，做出立體光暈感
            traces.append(go.Scatter(
                x=[row.rs_ratio], y=[row.rs_momentum],
                mode="markers",
                marker=dict(size=[size * 1.7], color=color, opacity=0.20,
                            symbol=symbol, line=dict(width=0)),
                showlegend=False, hoverinfo="skip",
            ))
            # 內層主體：實色、細金邊、文字標籤
            # 個股疊圖(highlight_tickers)用星形＋白色粗邊，跟板塊ETF的圓形區分開
            traces.append(go.Scatter(
                x=[row.rs_ratio], y=[row.rs_momentum],
                mode="markers+text",
                text=[row.label], textposition="top center",
                textfont=dict(color="rgba(240,230,200,0.9)", size=12),
                marker=dict(size=[size], color=color, opacity=0.92,
                            sizemode="diameter", symbol=symbol,
                            line=dict(width=2.4 if is_custom else 1.2,
                                      color="#FFFFFF" if is_custom else "rgba(212,175,55,0.85)")),
                showlegend=False,
                hovertemplate=f"<b>{row.label}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>",
            ))
        return traces

    fig = go.Figure()

    # 固定的圖例（跟動畫trace分開，避免象限切換時圖例跳動或重複）
    for q, color in QUADRANT_COLORS.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                  marker=dict(size=11, color=color), name=q))

    # 初始frame（第0週），接在圖例trace之後
    for tr in make_ticker_traces(0):
        fig.add_trace(tr)

    # 四象限底色與標籤（固定不隨動畫變動）
    fig.add_shape(type="rect", x0=100, x1=x_max, y0=100, y1=y_max,
                  fillcolor="rgba(63,167,150,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, x1=x_max, y0=y_min, y1=100,
                  fillcolor="rgba(212,175,55,0.06)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_min, x1=100, y0=y_min, y1=100,
                  fillcolor="rgba(192,67,90,0.08)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_min, x1=100, y0=100, y1=y_max,
                  fillcolor="rgba(91,141,190,0.08)", line_width=0, layer="below")
    fig.add_hline(y=100, line_color="rgba(212,175,55,0.35)", line_width=1)
    fig.add_vline(x=100, line_color="rgba(212,175,55,0.35)", line_width=1)
    for x, y, text in [
        (x_max * 0.999, y_max * 0.999, "領先 Leading"),
        (x_max * 0.999, y_min * 1.001, "轉弱 Weakening"),
        (x_min * 1.001, y_min * 1.001, "落後 Lagging"),
        (x_min * 1.001, y_max * 0.999, "改善 Improving"),
    ]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=False,
                            font=dict(size=12, color="rgba(212,175,55,0.55)"),
                            xanchor="right" if x > 100 else "left",
                            yanchor="top" if y > 100 else "bottom")

    # 建立每一週的frame。每支ETF佔2條trace(光暈+主體)，圖例trace固定在最前面不變。
    n_legend = len(QUADRANT_COLORS)
    ticker_trace_idx = list(range(n_legend, n_legend + len(ticker_order) * 2))
    frames = []
    for w in range(n_frames):
        frames.append(go.Frame(
            data=make_ticker_traces(w), name=str(w), traces=ticker_trace_idx,
        ))
    fig.frames = frames

    FRAME_MS = 550       # 每一幀的顯示間隔
    TWEEN_MS = 500       # 補間動畫時長，越接近FRAME_MS越滑順

    # 播放按鈕與時間滑桿
    fig.update_layout(
        paper_bgcolor=NAVY_BG,
        plot_bgcolor=NAVY_PANEL,
        font=dict(color="rgba(230,225,210,0.9)", family="Georgia, 'Noto Serif TC', serif"),
        xaxis_title="RS-Ratio（相對大盤強度，100為中性）",
        yaxis_title="RS-Momentum（強度變化動能，100為中性）",
        xaxis=dict(range=[x_min, x_max], gridcolor=GRID_LINE, zerolinecolor=GRID_LINE),
        yaxis=dict(range=[y_min, y_max], gridcolor=GRID_LINE, zerolinecolor=GRID_LINE),
        height=720,
        legend=dict(title="所在象限", bgcolor="rgba(0,0,0,0)",
                    font=dict(color="rgba(230,225,210,0.85)")),
        # b=130 留足夠空間給下方的播放鍵、滑桿、以及滑桿下面的日期文字，
        # 避免在手機窄螢幕上被切掉（之前 b=10 太小，日期文字會被裁掉）。
        margin=dict(t=30, b=130),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "x": 0.05, "y": -0.14, "xanchor": "left", "yanchor": "top",
            "buttons": [
                {"label": "▶ 播放", "method": "animate",
                 "args": [None, {"frame": {"duration": FRAME_MS, "redraw": False},
                                  "fromcurrent": True,
                                  "transition": {"duration": TWEEN_MS, "easing": "linear"}}]},
                {"label": "⏸ 暫停", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}]},
            ],
        }],
        sliders=[{
            "active": 0,
            "x": 0.15, "y": -0.14, "len": 0.8,
            "currentvalue": {"prefix": "", "visible": True,
                              "font": {"size": 13, "color": "rgba(230,225,210,0.9)"}},
            "pad": {"t": 20, "b": 10},
            "steps": [
                {"label": week_labels[w], "method": "animate",
                 "args": [[str(w)], {"frame": {"duration": FRAME_MS, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": TWEEN_MS, "easing": "linear"}}]}
                for w in range(n_frames)
            ],
        }],
    )
    return fig


# ----------------------------------------------------------------------
# 5. Streamlit 頁面
# ----------------------------------------------------------------------
st.title("美股板塊輪動 RRG（動畫版）")

_spy_daily, _spy_is_mock = fetch_daily_data([])
_market_health = compute_market_health(_spy_daily.get(BENCHMARK, pd.DataFrame()).get("close", pd.Series(dtype=float)))
if _market_health["ok"] is True:
    st.success(_market_health["label"])
elif _market_health["ok"] is False:
    st.error(_market_health["label"])
else:
    st.info(_market_health["label"])

tickers = list(SECTOR_ETFS.keys())

custom_input = st.text_input(
    "疊加你自己的持股(選填，用逗號分隔，例如 AAPL,TSLA)",
    value="",
    help="輸入的個股會用星形標記疊加在同一張圖上，跟你的持股比對，看它現在跟哪個板塊比較靠近、有沒有跟上輪動。",
)
custom_tickers = []
for t in custom_input.split(","):
    t = t.strip().upper()
    if t and t not in tickers and t not in custom_tickers:
        custom_tickers.append(t)

tickers_all = tickers + custom_tickers
prices, is_mock = fetch_price_volume_data(tickers_all)

_data_note = "目前為模擬資料。" if is_mock else "資料來源：Yahoo Finance（yfinance），每小時更新一次。"
st.caption(f"點下方「▶ 播放」看板塊如何隨時間在四象限間移動；泡泡大小＝資金流量（成交金額），資金湧入越多泡泡越大。{_data_note}")

range_label = st.selectbox("時間區間", list(RANGE_OPTIONS.keys()), index=3)
selected_frames = RANGE_OPTIONS[range_label]

rrg = compute_rrg_with_flow(prices, BENCHMARK)

# 個股代號打錯或資料不足時，跳過並提醒，不讓整頁掛掉
dropped = [t for t in custom_tickers if len(rrg.get(t, [])) < 5]
for t in dropped:
    st.warning(f"「{t}」的資料不足或代號可能有誤，暫不會顯示在圖上。")
    rrg.pop(t, None)
custom_tickers = [t for t in custom_tickers if t not in dropped]
ticker_order = [t for t in tickers_all if t in rrg]

long_df, n_frames, week_labels = build_animation_frame_data(
    rrg, SECTOR_ETFS, n_frames=selected_frames, highlight_tickers=set(custom_tickers),
)

fig = build_animated_figure(long_df, n_frames, ticker_order=ticker_order, week_labels=week_labels)
st.plotly_chart(fig, use_container_width=True)

alerts = compute_transition_alerts(long_df)
if alerts:
    lines = "\n".join(
        f"- **{t}**：{a['from']} → {a['to']}（照近期走勢推估，僅供留意，非預測）"
        for t, a in alerts.items()
    )
    st.warning(f"轉折警示：以下標的近期走勢，可能快要跨到不同象限\n\n{lines}")

streaks = compute_quadrant_streak(long_df)

with st.expander("查看目前(最新一週)各類股數值"):
    last = long_df[long_df["week_idx"] == long_df["week_idx"].max()].copy()
    last["象限停留週數"] = last["ticker"].map(streaks)
    show = last[["ticker", "rs_ratio", "rs_momentum", "dollar_flow", "quadrant", "象限停留週數"]].copy()
    show.columns = ["代號", "RS-Ratio", "RS-Momentum", "資金流量(估)", "所在象限", "象限停留週數"]
    st.dataframe(show.sort_values("RS-Ratio", ascending=False), use_container_width=True)
    st.caption("象限停留週數：目前這個象限已經連續待了幾週。數字越小代表剛換象限，通常是比較值得留意的轉折訊號。")

st.subheader("板塊成分股：Trend Template 篩選")
st.caption(
    "選一個板塊，看它的代表性重倉股裡有哪幾檔通過黑帶Trend Template的8條門檻。"
    "這份成分股清單是手動整理的代表性名單，不是即時官方持股；"
    "第8條相對強度(RS)是同組股票互相比較的近似排名，不是全市場RS Rating。"
)
drilldown_sector = st.selectbox(
    "選擇板塊", list(SECTOR_ETFS.keys()),
    format_func=lambda t: f"{t} {SECTOR_ETFS[t]}",
)
if st.button("查詢成分股 Trend Template"):
    constituents = SECTOR_CONSTITUENTS.get(drilldown_sector, [])
    with st.spinner("正在抓取成分股日線資料..."):
        daily_prices, daily_is_mock = fetch_daily_data(constituents)

    spy_close = daily_prices.get(BENCHMARK, pd.DataFrame()).get("close", pd.Series(dtype=float))
    closes = {t: daily_prices[t]["close"] for t in constituents
              if t in daily_prices and not daily_prices[t].empty}
    rs_pct = compute_rs_percentile(closes, spy_close)

    rows = []
    for t in constituents:
        cond = compute_trend_template(closes.get(t, pd.Series(dtype=float)))
        if cond is None:
            rows.append({"代號": t, "現價": "-", "均線結構(1-5)": "資料不足",
                         "股價位置(6-7)": "-", "RS百分位(近似)": "-", "8條全過": "-", "星級": "-"})
            continue
        ma_pass = cond["ma_structure_pass"]
        pos_pass = cond["price_position_pass"]
        rs = rs_pct.get(t)
        rs_pass = rs is not None and rs >= 70
        all_pass = ma_pass and pos_pass and rs_pass
        star = "★★☆☆" if all_pass else "★☆☆☆"
        rows.append({
            "代號": t,
            "現價": round(cond["price"], 2),
            "均線結構(1-5)": "通過" if ma_pass else "未過",
            "股價位置(6-7)": "通過" if pos_pass else "未過",
            "RS百分位(近似)": f"{rs}" if rs is not None else "資料不足",
            "8條全過": "✅" if all_pass else "—",
            "星級": star,
        })
    result_df = pd.DataFrame(rows)
    st.dataframe(result_df, use_container_width=True)
    st.caption(
        "星級只到★★☆☆（8條全過但VCP形態未自動判讀）——收縮型態偏主觀，"
        "建議把8條全過的標的丟進黑帶Trend Template/VCP skill，人工複核VCP階段再決定要不要盯Pivot。"
    )
    if daily_is_mock:
        st.caption("目前為模擬資料，數字僅供介面測試，非真實股價。")

st.info(
    "本工具僅彙整公開市場股價與成交量資訊做統計呈現，非投資顧問服務，"
    "不對特定標的提供分析意見或推介建議，亦不構成任何投資建議；"
    "投資人應自行判斷並自負盈虧。"
)
