"""
USBB 美股板塊輪動 · 四季輪轉
白貓黑帶師父浮世繪風格｜Streamlit + yfinance + Plotly
"""
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------- 設定
BENCH = "SPY"
SECTORS = {
    "XLK": "科技", "XLC": "通訊服務", "XLY": "非必需消費", "XLF": "金融",
    "XLI": "工業", "XLV": "醫療保健", "XLE": "能源", "XLB": "原物料",
    "XLP": "必需消費", "XLU": "公用事業", "XLRE": "房地產",
}
# 浮世繪色系，11 色容易分辨
COLORS = {
    "XLK": "#1A3A52", "XLC": "#6B4C9A", "XLY": "#C0392B", "XLF": "#2E7D6B",
    "XLI": "#7A4E2D", "XLV": "#D4679A", "XLE": "#E08E0B", "XLB": "#5B7F2B",
    "XLP": "#3D7CB8", "XLU": "#8A6D12", "XLRE": "#555555",
}
MONTHS_BACK = 36          # 可查詢三年
WINDOW = 14               # RS-Ratio 平滑週期（週）
MOM_WINDOW = 6            # RS-Momentum 平滑週期（週）

SEASONS = {
    "spring": {"name": "春", "label": "改善", "desc": "相對強度還在大盤之下，但動能已經回升。像冬雪剛化、花苞初開，值得開始留意。", "color": "#F6D6DE", "ink": "#B0476A", "img": "spring", "pos": "50% 45%"},
    "summer": {"name": "夏", "label": "領先", "desc": "強度和動能都贏過大盤，資金正在這裡。領頭股最常出現在這一季。", "color": "#DCEBD3", "ink": "#2F7D55", "img": "summer", "pos": "50% 12%"},
    "autumn": {"name": "秋", "label": "轉弱", "desc": "強度還在大盤之上，但動能開始下降。高峰已過，持有的人要提高警覺。", "color": "#F6DDC8", "ink": "#B5541C", "img": "autumn", "pos": "50% 18%"},
    "winter": {"name": "冬", "label": "落後", "desc": "強度和動能都輸給大盤。休養生息的階段，等待春天回來。", "color": "#DDE6EE", "ink": "#46647E", "img": "winter", "pos": "50% 35%"},
}
ORDER = ["spring", "summer", "autumn", "winter"]

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="美股板塊輪動｜AI美股黑帶學院", page_icon="🐈", layout="wide")


# ---------------------------------------------------------------- 圖片
def img_uri(name: str):
    for ext in ("webp", "png", "jpg", "jpeg"):
        p = ASSETS / f"{name}.{ext}"
        if p.exists():
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            return f"data:image/{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    return None


# ---------------------------------------------------------------- 樣式
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+TC:wght@600;900&family=Noto+Sans+TC:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--navy:#1A3A52; --gold:#D4AF37; --paper:#F5F3EC; --ink:#1E2A33; --muted:#5B6B77}
html, body, [class*="css"], .stApp{font-family:"Noto Sans TC","PingFang TC","Microsoft JhengHei",sans-serif}
.stApp{background:var(--paper)}
h1,h2,h3,.serif{font-family:"Noto Serif TC","Songti TC",serif !important; color:var(--navy)}
.block-container{padding-top:1.2rem; max-width:1280px}
.banner{position:relative; border-radius:6px; overflow:hidden; margin-bottom:1.2rem; background:var(--navy); display:grid; grid-template-columns:minmax(0,5fr) minmax(0,7fr); align-items:center; min-height:300px}
.banner img{grid-column:2; grid-row:1; display:block; width:100%; height:300px; object-fit:cover; object-position:50% 55%; -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 28%); mask-image:linear-gradient(90deg,transparent 0,#000 28%)}
.banner .txt{grid-column:1; grid-row:1; position:relative; z-index:1; padding:28px 0 28px 36px; margin-right:-8%; color:#fff}
.banner h1{color:#fff !important; font-size:clamp(26px,3.2vw,40px); margin:0 0 10px; padding:0; line-height:1.3}
.banner p{margin:0; color:#E6EDF2; font-size:16px; line-height:1.7}
.banner.noimg{display:block; min-height:0} .banner.noimg .txt{margin:0; padding:28px 36px}
@media (max-width:760px){
  .banner{display:flex; flex-direction:column; min-height:0}
  .banner .txt{order:-1; margin:0; padding:24px 20px 0}
  .banner img{height:220px; margin-top:-30px; -webkit-mask-image:linear-gradient(180deg,transparent 0,#000 30%); mask-image:linear-gradient(180deg,transparent 0,#000 30%)}
}
.season{border-radius:6px; overflow:hidden; background:#fff; border:1px solid #E3DED0; height:100%}
.season img{display:block; width:100%; aspect-ratio:4/3; object-fit:cover}
.season .hd{padding:12px 16px 4px; font-family:"Noto Serif TC",serif; font-weight:900; font-size:22px}
.season .hd small{font-family:"Noto Sans TC",sans-serif; font-weight:500; font-size:14px; margin-left:6px; opacity:.8}
.season .ds{padding:0 16px 8px; font-size:13.5px; color:var(--muted); line-height:1.6}
.season ul{list-style:none; margin:0; padding:4px 16px 14px}
.season li{display:flex; justify-content:space-between; gap:8px; padding:6px 0; border-top:1px dashed #E3DED0; font-size:15px}
.season li .mv{font-size:13px; color:var(--muted); white-space:nowrap}
.season li.new .mv{color:var(--ink); font-weight:700}
.season .empty{padding:6px 16px 16px; font-size:14px; color:var(--muted)}
.dot{display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; vertical-align:1px}
.note{font-size:13px; color:var(--muted)}
.stTabs [data-baseweb="tab"]{font-family:"Noto Serif TC",serif; font-weight:600; font-size:16px}
footer, #MainMenu{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- 資料
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_prices() -> pd.DataFrame:
    tickers = list(SECTORS) + [BENCH]
    raw = yf.download(tickers, period="5y", interval="1wk", auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    return close.dropna(how="all").ffill()


def compute_rrg(close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """JdK RRG 近似算法：相對強度 → 平滑比率 → 動能。以 100 為中心。"""
    bench = close[BENCH]
    ratio, mom = {}, {}
    for t in SECTORS:
        if t not in close:
            continue
        rs = 100 * close[t] / bench
        r = 100 * rs.ewm(span=WINDOW).mean() / rs.rolling(WINDOW * 2).mean()
        m = 100 * r / r.rolling(MOM_WINDOW).mean()
        ratio[t], mom[t] = r, m
    ratio, mom = pd.DataFrame(ratio), pd.DataFrame(mom)
    ok = ratio.notna().all(axis=1) & mom.notna().all(axis=1)
    return ratio[ok], mom[ok]


def season_of(r: float, m: float) -> str:
    if r >= 100 and m >= 100:
        return "summer"
    if r >= 100:
        return "autumn"
    if m >= 100:
        return "spring"
    return "winter"


def month_ends(index: pd.DatetimeIndex) -> pd.Series:
    """每個月最後一個週資料點。"""
    s = pd.Series(index, index=index)
    return s.groupby(index.to_period("M")).max()


# ---------------------------------------------------------------- 圖表
def quadrant_shapes(xr, yr):
    x0, x1 = xr
    y0, y1 = yr
    boxes = [("spring", x0, 100, 100, y1), ("summer", 100, x1, 100, y1),
             ("autumn", 100, x1, y0, 100), ("winter", x0, 100, y0, 100)]
    shapes, notes = [], []
    for key, a, b, c, d in boxes:
        s = SEASONS[key]
        shapes.append(dict(type="rect", x0=a, x1=b, y0=c, y1=d, fillcolor=s["color"], opacity=.55, line_width=0, layer="below"))
        notes.append(dict(x=a + (b - a) * (0.06 if a < 100 else 0.94), y=c + (d - c) * (0.93 if c >= 100 else 0.07),
                          text=f"<b>{s['name']}</b>　{s['label']}", showarrow=False, font=dict(size=15, color=s["ink"], family="Noto Serif TC, serif"),
                          xanchor="left" if a < 100 else "right"))
    shapes += [dict(type="line", x0=100, x1=100, y0=y0, y1=y1, line=dict(color="#1A3A52", width=1, dash="dot")),
               dict(type="line", x0=x0, x1=x1, y0=100, y1=100, line=dict(color="#1A3A52", width=1, dash="dot"))]
    return shapes, notes


def axis_range(ratio, mom, cols):
    pad = .4
    xs, ys = ratio[cols].values, mom[cols].values
    xr = max(abs(np.nanmin(xs) - 100), abs(np.nanmax(xs) - 100)) + pad
    yr = max(abs(np.nanmin(ys) - 100), abs(np.nanmax(ys) - 100)) + pad
    return (100 - xr, 100 + xr), (100 - yr, 100 + yr)


def base_layout(xr, yr, height=620):
    shapes, notes = quadrant_shapes(xr, yr)
    return dict(
        height=height, shapes=shapes, annotations=notes, showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBFAF5", margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(range=xr, title="相對強度（右邊 = 比大盤強）", zeroline=False, gridcolor="#ECE7DA"),
        yaxis=dict(range=yr, title="相對動能（上面 = 越來越強）", zeroline=False, gridcolor="#ECE7DA"),
        font=dict(family="Noto Sans TC, sans-serif", color="#1E2A33"),
        hoverlabel=dict(bgcolor="#fff", font_size=13),
    )


def traces_for(ratio, mom, end, tickers, tail):
    idx = ratio.index.get_loc(end)
    sl = slice(max(0, idx - tail + 1), idx + 1)
    out = []
    for t in tickers:
        xs, ys = ratio[t].iloc[sl], mom[t].iloc[sl]
        c = COLORS[t]
        out.append(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=c, width=2, shape="spline"),
                              marker=dict(size=[4] * (len(xs) - 1) + [0], color=c), hoverinfo="skip"))
        out.append(go.Scatter(x=[xs.iloc[-1]], y=[ys.iloc[-1]], mode="markers+text",
                              marker=dict(size=14, color=c, line=dict(color="#fff", width=2)),
                              text=[SECTORS[t]], textposition="top center", textfont=dict(size=13, color=c),
                              hovertemplate=f"<b>{SECTORS[t]} {t}</b><br>強度 %{{x:.2f}}<br>動能 %{{y:.2f}}<extra></extra>"))
    return out


# ---------------------------------------------------------------- 版面
banner = img_uri("banner")
st.markdown(f"""
<div class="banner{'' if banner else ' noimg'}">
  {f'<img src="{banner}" alt="白貓師父站在四季輪轉的石庭中央">' if banner else ''}
  <div class="txt">
    <h1>美股板塊輪動・四季輪轉</h1>
    <p>資金像四季一樣在板塊之間輪流移動。選一個月份，看十一個板塊那時各自走到哪一季，也看它們是從哪裡轉過來的。</p>
  </div>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner("正在讀取最新的板塊資料…"):
        close = load_prices()
        ratio, mom = compute_rrg(close)
except Exception:
    st.error("暫時讀不到行情資料，可能是資料來源忙線。請過幾分鐘重新整理頁面。")
    st.stop()

if ratio.empty:
    st.error("資料不足，無法計算輪動位置。請稍後再試。")
    st.stop()

ends = month_ends(ratio.index).iloc[-MONTHS_BACK:]
labels = {p.strftime("%Y 年 %m 月"): d for p, d in ends.items()}
label_list = list(labels)

c1, c2, c3 = st.columns([2, 3, 2])
with c1:
    pick = st.selectbox("查詢月份", label_list[::-1], index=0)
with c2:
    chosen = st.multiselect("顯示板塊", list(SECTORS), default=list(SECTORS),
                            format_func=lambda t: f"{SECTORS[t]} {t}")
with c3:
    tail = st.slider("軌跡長度（週）", 2, 12, 6)

if not chosen:
    st.info("至少選一個板塊，圖上才會有東西。")
    st.stop()

end = labels[pick]
pos = label_list.index(pick)
prev_end = labels[label_list[pos - 1]] if pos > 0 else None
last_date = ratio.index[-1]
st.markdown(f'<p class="note">資料計算到 {end:%Y/%m/%d} 當週｜比較基準：S&P 500（SPY）｜最新資料：{last_date:%Y/%m/%d}</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["這個月的輪動", "播放三年輪動", "單一板塊的四季歷程"])

# ---- 分頁一：當月
with tab1:
    xr, yr = axis_range(ratio.loc[:end].iloc[-tail:], mom.loc[:end].iloc[-tail:], chosen)
    fig = go.Figure(traces_for(ratio, mom, end, chosen, tail), layout=base_layout(xr, yr))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    now = {t: season_of(ratio.at[end, t], mom.at[end, t]) for t in chosen}
    before = {t: season_of(ratio.at[prev_end, t], mom.at[prev_end, t]) for t in chosen} if prev_end is not None else {}

    st.markdown('<h3 class="serif">這個月，各板塊在哪一季</h3>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, key in zip(cols, ORDER):
        s = SEASONS[key]
        im = img_uri(s["img"])
        items = []
        for t in [t for t in chosen if now[t] == key]:
            b = before.get(t)
            moved = b is not None and b != key
            mv = f"{SEASONS[b]['name']} → {s['name']}" if moved else ("持續" if b else "")
            items.append(f'<li class="{"new" if moved else ""}"><span><span class="dot" style="background:{COLORS[t]}"></span>{SECTORS[t]}</span><span class="mv">{mv}</span></li>')
        body = f"<ul>{''.join(items)}</ul>" if items else '<div class="empty">這個月沒有板塊在這一季。</div>'
        pic = '<img src="{}" style="object-position:{}" alt="{}季的白貓師父">'.format(im, s["pos"], s["name"]) if im else ""
        col.markdown(f"""<div class="season">{pic}
            <div class="hd" style="color:{s['ink']}">{s['name']}<small>{s['label']}</small></div>
            <div class="ds">{s['desc']}</div>{body}</div>""", unsafe_allow_html=True)

    moved = [t for t in chosen if before and before.get(t) != now[t]]
    if moved:
        txt = "、".join(f"{SECTORS[t]}（{SEASONS[before[t]]['name']} → {SEASONS[now[t]]['name']}）" for t in moved)
        st.markdown(f"**跟上個月比，換季的板塊：** {txt}")
    elif before:
        st.markdown("**跟上個月比，沒有板塊換季。**")

# ---- 分頁二：動畫
with tab2:
    st.markdown('<p class="note">按下播放，一個月一個月看資金怎麼在板塊之間移動。軌跡長度跟上方設定一樣。</p>', unsafe_allow_html=True)
    ends_list = list(labels.values())
    first_i = ratio.index.get_loc(ends_list[0])
    xr, yr = axis_range(ratio.iloc[max(0, first_i - tail):], mom.iloc[max(0, first_i - tail):], chosen)
    frames = [go.Frame(data=traces_for(ratio, mom, d, chosen, tail), name=lbl) for lbl, d in labels.items()]
    lay = base_layout(xr, yr, height=660)
    lay["updatemenus"] = [dict(type="buttons", x=0, y=-0.1, xanchor="left", showactive=False, direction="left",
                               buttons=[dict(label="▶ 播放", method="animate", args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True, transition=dict(duration=300))]),
                                        dict(label="❚❚ 暫停", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])]
    lay["sliders"] = [dict(active=0, x=0.18, len=0.82, y=-0.06, currentvalue=dict(prefix="", font=dict(size=16, family="Noto Serif TC, serif")),
                           steps=[dict(label=l, method="animate", args=[[l], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for l in labels])]
    lay["margin"]["b"] = 110
    fig2 = go.Figure(data=frames[0].data, frames=frames, layout=lay)
    st.plotly_chart(fig2, width="stretch", config={"displayModeBar": False})

# ---- 分頁三：單一板塊
with tab3:
    t = st.selectbox("選擇板塊", list(SECTORS), format_func=lambda t: f"{SECTORS[t]} {t}")
    seq = [(lbl, season_of(ratio.at[d, t], mom.at[d, t])) for lbl, d in labels.items()]
    cells = "".join(
        f'<div title="{lbl}：{SEASONS[k]["name"]}・{SEASONS[k]["label"]}" style="flex:1; min-width:14px; height:42px; background:{SEASONS[k]["color"]}; border-right:1px solid #fff; display:flex; align-items:center; justify-content:center; font-family:Noto Serif TC,serif; font-weight:900; color:{SEASONS[k]["ink"]}; font-size:13px">{SEASONS[k]["name"]}</div>'
        for lbl, k in seq)
    st.markdown(f'<h3 class="serif">{SECTORS[t]}（{t}）過去三年每個月的季節</h3>'
                f'<div style="display:flex; border-radius:4px; overflow:hidden; border:1px solid #E3DED0">{cells}</div>'
                f'<div style="display:flex; justify-content:space-between" class="note"><span>{label_list[0]}</span><span>{label_list[-1]}</span></div>',
                unsafe_allow_html=True)

    counts = pd.Series([k for _, k in seq]).value_counts()
    st.markdown("　".join(f"{SEASONS[k]['name']}：{counts.get(k, 0)} 個月" for k in ORDER))

    changes = [f"{seq[i][0]}　{SEASONS[seq[i-1][1]]['name']} → {SEASONS[seq[i][1]]['name']}" for i in range(1, len(seq)) if seq[i][1] != seq[i - 1][1]]
    with st.expander(f"換季紀錄（共 {len(changes)} 次）"):
        st.markdown("\n".join(f"- {c}" for c in changes[::-1]) if changes else "這段期間沒有換季。")

    s = ratio[t].loc[ends_list[0] - pd.Timedelta(weeks=tail):]
    m = mom[t].loc[s.index]
    fig3 = go.Figure()
    fig3.add_scatter(x=s.index, y=s, name="相對強度", line=dict(color="#1A3A52", width=2))
    fig3.add_scatter(x=m.index, y=m, name="相對動能", line=dict(color="#D4AF37", width=2))
    fig3.add_hline(y=100, line_dash="dot", line_color="#999")
    fig3.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FBFAF5", margin=dict(l=40, r=20, t=10, b=30),
                       legend=dict(orientation="h", y=1.1), font=dict(family="Noto Sans TC, sans-serif"),
                       yaxis=dict(gridcolor="#ECE7DA"), xaxis=dict(gridcolor="#ECE7DA"))
    st.plotly_chart(fig3, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------------------- 說明
with st.expander("怎麼看這張圖"):
    st.markdown("""
- 每個點是一個板塊，拿它跟 S&P 500 比。越往右，代表比大盤越強；越往上，代表強勢正在加速。
- 板塊通常照 **春 → 夏 → 秋 → 冬** 的順時針方向輪轉，就像四季。
- 尾巴是它過去幾週走過的路，看得出它正要往哪一季走。
- 圖只告訴你資金大方向。板塊在夏季，不代表裡面每一檔都會漲；要不要買、什麼時候買，還是要回到個股的看盤和進場紀律。
- 計算方式為 RRG（相對輪動圖）的近似算法，以週資料計算，數值與其他平台可能略有差異。
""")

st.markdown("""
<div style="margin-top:2rem; padding:24px 28px; background:#1A3A52; border-radius:6px; color:#fff; display:flex; flex-wrap:wrap; gap:16px; align-items:center; justify-content:space-between">
  <div><div style="font-family:'Noto Serif TC',serif; font-weight:900; font-size:22px">知道風往哪裡吹之後，下一步是找股票</div>
  <div style="color:#C9D6DF; font-size:15px; margin-top:4px">強勢板塊裡，哪幾檔才值得研究？到第 2 站綠帶繼續練。</div></div>
  <a href="https://usbbtrading.com/tools#s2" target="_blank" style="background:#D4AF37; color:#122B3D; font-weight:700; padding:12px 22px; border-radius:4px; text-decoration:none">下一站：找出值得研究的股票</a>
</div>
""", unsafe_allow_html=True)

st.markdown('<p class="note" style="margin-top:2rem; text-align:center">AI 美股黑帶學院 US Stock Black Belt (USBB)　本工具僅供教育用途，不構成任何投資建議。投資有風險，請自行判斷並承擔盈虧。</p>', unsafe_allow_html=True)
