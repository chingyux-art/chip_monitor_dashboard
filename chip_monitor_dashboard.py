import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import re

# =========================
# 1) Page config
# =========================
st.set_page_config(
    page_title="台股籌碼監控儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 避免長時間抓資料時前端連線超時（某些環境會需要）
st.set_option('server.httpTimeout', 300)

# CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive { color: #d32f2f; }
    .negative { color: #1976d2; }
</style>
""", unsafe_allow_html=True)

st.title("📊 台股籌碼監控儀表板 v2.0（最近10個開市日）")
st.markdown("---")

# =========================
# 2) Sidebar
# =========================
st.sidebar.header("⚙️ 設定")
data_source = st.sidebar.radio(
    "選擇數據源",
    options=["真實數據 (TWSE)", "模擬數據 (測試用)"],
    help="真實數據來自台灣證券交易所"
)

# 監控的20支股票
STOCKS_LIST = [
    ('2330', '台積電'),
    ('2454', '聯發科'),
    ('2317', '鴻海'),
    ('2412', '中華電'),
    ('1101', '台泥'),
    ('2881', '富邦金'),
    ('2882', '國泰金'),
    ('2886', '兆豐金'),
    ('2891', '中信金'),
    ('2892', '第一金'),
    ('2883', '開發金'),
    ('2884', '玉山金'),
    ('2885', '元大金'),
    ('2880', '華南金'),
    ('1303', '南亞'),
    ('1301', '台塑'),
    ('1326', '台化'),
    ('2409', '友達'),
    ('2408', '奇美電'),
    ('3034', '聯詠'),
]

WATCH_CODES = [c for c, _ in STOCKS_LIST]
CODE_TO_NAME = {c: n for c, n in STOCKS_LIST}

# =========================
# 3) HTTP session with retries
# =========================
def create_session_with_retries():
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session

# =========================
# 4) Helpers: parse, column finding
# =========================
def to_int(x):
    """安全把帶逗號/空值/-- 的字串轉 int"""
    if x is None:
        return 0
    s = str(x).strip()
    if s in ("", "--", "—", "NaN", "nan", "None"):
        return 0
    # 移除逗號、空白、特殊符號
    s = s.replace(",", "").replace(" ", "")
    # 若有括號負號等情況
    s = s.replace("(", "-").replace(")", "")
    # 只保留數字與負號
    m = re.findall(r"-?\d+", s)
    return int(m[0]) if m else 0

def find_col(cols, keywords):
    """在欄位名稱中找包含 keywords 任一字串的欄位（回傳第一個命中）"""
    for k in keywords:
        for c in cols:
            if k in str(c):
                return c
    return None

# =========================
# 5) 精準最近 N 個「有交易」開市日：用 T86 回傳是否有 data 判斷
# =========================
@st.cache_data(ttl=3600)
def get_last_n_twse_open_days(n=10, max_lookback=45):
    """
    精準抓最近 n 個實際有交易的開市日：
    - 逐日呼叫 TWSE T86
    - 只要回傳 data 非空且 stat=OK，視為開市日
    - 先跳過週末，減少無效請求
    """
    session = create_session_with_retries()

    def fetch_t86_json(date_str: str):
        # 改用 rwd/zh 版（常見可用範例）[3](https://github.com/TsengYuanChe/python_stock/blob/main/TWSE.py)[4](https://github.com/arleigh418/python-and-Taiwan-stock-market/issues/76)
        url = (
            "https://www.twse.com.tw/rwd/zh/fund/T86"
            f"?date={date_str}&selectType=ALLBUT0999&response=json"
        )
        r = session.get(url, timeout=(3, 8))  # (connect, read)
        r.raise_for_status()
        return r.json()

    open_days = []
    d = datetime.now().date()
    looked = 0

    while len(open_days) < n and looked < max_lookback:
        # 先跳過週末，避免白打
        if d.weekday() >= 5:
            d -= timedelta(days=1)
            looked += 1
            continue

        date_str = d.strftime("%Y%m%d")
        try:
            js = fetch_t86_json(date_str)
            # T86 典型回傳包含 stat/fields/data [5](https://www.twse.com.tw/fund/T86)
            if str(js.get("stat", "")).upper() == "OK" and js.get("data"):
                open_days.append(d)
        except Exception:
            pass

        d -= timedelta(days=1)
        looked += 1
        time.sleep(0.03)  # 稍微降速，避免被限流

    open_days = sorted(open_days)
    if not open_days:
        return [], None, None
    return open_days, open_days[0], open_days[-1]

# =========================
# 6) 以「每個開市日抓一次」，再篩 20 檔（效能比逐檔逐日好很多）
# =========================
@st.cache_data(ttl=3600)
def fetch_t86_for_dates(open_days):
    """
    針對指定 open_days（list[date]），逐日抓 T86，回傳合併 DataFrame
    """
    session = create_session_with_retries()
    frames = []

    for d in open_days:
        date_str = d.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/fund/T86?response=json&date={date_str}&selectType=ALLBUT0999"
        try:
            r = session.get(url, timeout=10)
            if r.status_code != 200:
                continue
            js = r.json()
            if not js.get("data"):
                continue

            df = pd.DataFrame(js["data"], columns=js.get("fields", []))
            df["日期"] = pd.to_datetime(d)
            frames.append(df)
        except Exception:
            continue

        time.sleep(0.05)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def normalize_t86_to_app_schema(t86_df):
    """
    把 T86 原始欄位轉成你 app 用的欄位：
    日期、股票代碼、股票名稱、外資買超、內資買超(投信)、自營商買超
    """
    if t86_df.empty:
        return pd.DataFrame()

    cols = list(t86_df.columns)

    code_col = find_col(cols, ["證券代號", "股票代碼", "代號"])
    name_col = find_col(cols, ["證券名稱", "股票名稱", "名稱"])

    foreign_col = find_col(cols, ["外陸資買賣超股數", "外資及陸資買賣超股數", "外資買賣超股數"])
    trust_col = find_col(cols, ["投信買賣超股數", "投信買賣超"])
    dealer_total_col = find_col(cols, ["自營商買賣超股數", "自營商買賣超"])
    dealer_self_col = find_col(cols, ["自營商買賣超股數(自行買賣)", "自營商買賣超(自行買賣)"])
    dealer_hedge_col = find_col(cols, ["自營商買賣超股數(避險)", "自營商買賣超(避險)"])

    if code_col is None:
        # 若連代號都找不到，無法用
        return pd.DataFrame()

    out = pd.DataFrame()
    out["日期"] = pd.to_datetime(t86_df["日期"])
    out["股票代碼"] = t86_df[code_col].astype(str).str.strip()
    out["股票名稱"] = t86_df[name_col].astype(str).str.strip() if name_col else out["股票代碼"]

    # 外資
    if foreign_col:
        out["外資買超"] = t86_df[foreign_col].apply(to_int)
    else:
        out["外資買超"] = 0

    # 內資（這裡用投信代表內資）
    if trust_col:
        out["內資買超"] = t86_df[trust_col].apply(to_int)
    else:
        out["內資買超"] = 0

    # 自營商
    if dealer_total_col:
        out["自營商買超"] = t86_df[dealer_total_col].apply(to_int)
    else:
        # 若沒有總自營商，嘗試用 自行買賣 + 避險 相加
        if dealer_self_col or dealer_hedge_col:
            s = t86_df[dealer_self_col].apply(to_int) if dealer_self_col else 0
            h = t86_df[dealer_hedge_col].apply(to_int) if dealer_hedge_col else 0
            out["自營商買超"] = s + h
        else:
            out["自營商買超"] = 0

    # 只保留你監控清單的 20 檔
    out = out[out["股票代碼"].isin(WATCH_CODES)].copy()

    # 若名稱缺失，用你清單補齊
    out["股票名稱"] = out["股票代碼"].map(CODE_TO_NAME).fillna(out["股票名稱"])

    return out

# =========================
# 7) 模擬資料
# =========================
@st.cache_data(ttl=3600)
def generate_institutional_data(open_days, stocks_list):
    np.random.seed(42)
    dates = pd.to_datetime(open_days)
    data = []
    for stock_code, stock_name in stocks_list:
        for date in dates:
            foreign_buy = np.random.randint(-500, 500)
            domestic_buy = np.random.randint(-300, 300)
            dealer_buy = np.random.randint(-200, 200)

            data.append({
                "股票代碼": stock_code,
                "股票名稱": stock_name,
                "日期": date,
                "外資買超": foreign_buy,
                "內資買超": domestic_buy,
                "自營商買超": dealer_buy,
            })
    return pd.DataFrame(data)

# =========================
# 8) streak / score / kline
# =========================
def calculate_buy_streaks(df):
    out = []
    for code in df["股票代碼"].unique():
        s = df[df["股票代碼"] == code].sort_values("日期").copy()
        s["總買超"] = s["外資買超"] + s["內資買超"] + s["自營商買超"]

        consec = 0
        max_streak = 0
        for v in s["總買超"]:
            if v > 0:
                consec += 1
                max_streak = max(max_streak, consec)
            else:
                consec = 0

        current_streak = 0
        for v in reversed(s["總買超"].tolist()):
            if v > 0:
                current_streak += 1
            else:
                break

        name = s["股票名稱"].iloc[0] if "股票名稱" in s.columns and len(s) else code
        out.append({
            "股票代碼": code,
            "股票名稱": name,
            "最大連買天數": max_streak,
            "連買天數": current_streak
        })
    return pd.DataFrame(out)

def calculate_score(row):
    """
    0-100 綜合評分（保留你原本權重；真實數據可能尺度很大，容易飽和）
    """
    score = 50
    score += (row["外資買超"] / 10) * 0.3
    score += (row["內資買超"] / 10) * 0.2
    score += (row["自營商買超"] / 10) * 0.2
    score += (row["連買天數"] / 5) * 0.3
    return max(0, min(100, score))

@st.cache_data(ttl=3600)
def get_kline_data(stock_code, start, end):
    for suffix in [".TW", ".TWO"]:
        try:
            ticker = f"{stock_code}{suffix}"
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df is not None and len(df) > 0:
                return df
        except Exception:
            pass
    return None

# =========================
# 9) 主流程：先算最近 10 開市日，再載入資料
# =========================
# 先顯示基本 UI，避免一進來就是空白等待
status_box = st.empty()

with st.spinner("正在取得最近 10 個 TWSE 開市日..."):
    open_days, start_date, end_date = get_last_n_twse_open_days(n=10, max_lookback=45)

if not open_days:
    status_box.error("⚠️ 無法取得最近開市日（TWSE 可能暫時無回應）。請切換到模擬數據或稍後再試。")
    data_source = "模擬數據 (測試用)"
    # 退而求其次：用最近10個平日（非精準）
    approx = []
    d = datetime.now().date()
    while len(approx) < 10:
        if d.weekday() < 5:
            approx.append(d)
        d -= timedelta(days=1)
    open_days = sorted(approx)
    start_date, end_date = open_days[0], open_days[-1]
else:
    status_box.success(f"✅ 已取得最近10個開市日：{start_date} ～ {end_date}")

# 添加總買超列
institutional_df["總買超"] = (
    institutional_df["外資買超"] +
    institutional_df["內資買超"] +
    institutional_df["自營商買超"]
)

# 計算連買天數
consecutive_buying_df = calculate_buy_streaks(institutional_df)

# 匯總最新數據（以 end_date 那天的資料為準，若缺就用該股最後一筆）
latest_data = []
for stock_code, stock_name in STOCKS_LIST:
    stock_data = institutional_df[institutional_df["股票代碼"] == stock_code].sort_values("日期")
    if len(stock_data) == 0:
        continue

    latest = stock_data.iloc[-1]
    consecutive = consecutive_buying_df[consecutive_buying_df["股票代碼"] == stock_code]["連買天數"].values

    row_data = {
        "股票代碼": stock_code,
        "股票名稱": stock_name,
        "外資買超": int(latest["外資買超"]),
        "內資買超": int(latest["內資買超"]),
        "自營商買超": int(latest["自營商買超"]),
        "總買超": int(latest["總買超"]),
        "連買天數": int(consecutive[0]) if len(consecutive) > 0 else 0
    }
    row_data["評分"] = calculate_score(row_data)
    latest_data.append(row_data)

summary_df = pd.DataFrame(latest_data)

# 若 summary_df 意外為空，避免後面 tabs 爆掉
if summary_df.empty:
    st.error("⚠️ summary_df 為空，無法顯示圖表。請切換數據源或稍後再試。")
    st.stop()

# =========================
# 10) Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["法人籌碼", "熱門股票", "K線分析", "詳細資料"])

# ---- Tab 1 ----
with tab1:
    st.header("📈 法人籌碼分析")

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_foreign = summary_df["外資買超"].mean()
        st.metric("平均外資買超", f"{avg_foreign:.0f}", f"{avg_foreign:+.0f}" if avg_foreign > 0 else f"{avg_foreign:.0f}")
    with col2:
        avg_domestic = summary_df["內資買超"].mean()
        st.metric("平均內資買超", f"{avg_domestic:.0f}", f"{avg_domestic:+.0f}" if avg_domestic > 0 else f"{avg_domestic:.0f}")
    with col3:
        avg_dealer = summary_df["自營商買超"].mean()
        st.metric("平均自營商買超", f"{avg_dealer:.0f}", f"{avg_dealer:+.0f}" if avg_dealer > 0 else f"{avg_dealer:.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(data=[
            go.Bar(x=summary_df["股票名稱"], y=summary_df["外資買超"], name="外資"),
            go.Bar(x=summary_df["股票名稱"], y=summary_df["內資買超"], name="內資(投信)"),
            go.Bar(x=summary_df["股票名稱"], y=summary_df["自營商買超"], name="自營商")
        ])
        fig.update_layout(title="各法人買賣超對比", barmode="group", height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if len(consecutive_buying_df) > 0:
            top_consecutive = consecutive_buying_df.nlargest(10, "連買天數")
            fig = px.bar(top_consecutive, x="股票名稱", y="連買天數", color="連買天數",
                         title="連買天數 Top 10", color_continuous_scale="Viridis")
            fig.update_layout(height=400, hovermode="x")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("最新法人籌碼數據")
    display_df = summary_df[["股票代碼", "股票名稱", "外資買超", "內資買超", "自營商買超", "總買超"]].copy()
    for c in ["外資買超", "內資買超", "自營商買超", "總買超"]:
        display_df[c] = display_df[c].apply(lambda x: f"{int(x):+d}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---- Tab 2 ----
with tab2:
    st.header("🔥 熱門股票排行")

    top_20 = summary_df.nlargest(20, "評分").reset_index(drop=True)
    top_20["排名"] = range(1, len(top_20) + 1)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("評分排行榜 Top 20")
        display_top20 = top_20[["排名", "股票代碼", "股票名稱", "評分", "連買天數", "總買超"]].copy()
        display_top20["評分"] = display_top20["評分"].apply(lambda x: f"{x:.1f}")
        display_top20["總買超"] = display_top20["總買超"].apply(lambda x: f"{int(x):+d}")
        st.dataframe(display_top20, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("評分分布")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=top_20["評分"], nbinsx=15, name="評分",
                                   marker_color="rgb(55, 83, 109)", opacity=0.7))
        fig.update_layout(title="評分分布", xaxis_title="評分 (0-100)", yaxis_title="股票數量",
                          height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("評分 vs 連買天數")
    top_20_plot = top_20.copy()
    top_20_plot["BubbleSize"] = top_20_plot["總買超"].abs()
    fig = px.scatter(
        top_20_plot,
        x="連買天數",
        y="評分",
        size="BubbleSize",
        hover_name="股票名稱",
        color="評分",
        color_continuous_scale="RdYlGn",
        title="評分 vs 連買天數分析",
        size_max=40
    )
    fig.update_layout(height=500, hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3 ----
with tab3:
    st.header("📈 K線分析")

    default_choice = []
    if not summary_df.empty:
        default_choice = [f"{summary_df.iloc[0]['股票代碼']} {summary_df.iloc[0]['股票名稱']}"]

    selected_stocks = st.multiselect(
        "選擇股票",
        options=[f"{row['股票代碼']} {row['股票名稱']}" for _, row in summary_df.iterrows()],
        default=default_choice,
        max_selections=3
    )

    if selected_stocks:
        for selected in selected_stocks:
            stock_code_display = selected.split()[0]
            stock_name = selected.split()[1]

            st.subheader(f"{stock_name} ({stock_code_display})")

            kline_data = get_kline_data(stock_code_display, start_date, end_date + timedelta(days=1))

            if kline_data is not None and len(kline_data) > 0:
                fig = go.Figure(data=[go.Candlestick(
                    x=kline_data.index,
                    open=kline_data["Open"],
                    high=kline_data["High"],
                    low=kline_data["Low"],
                    close=kline_data["Close"],
                    name=stock_name
                )])
                fig.update_layout(title=f"{stock_name} K線圖", yaxis_title="股價 (TWD)",
                                  xaxis_title="日期", template="plotly_white",
                                  height=500, hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("最高價", f"${kline_data['High'].max():.2f}")
                with col2:
                    st.metric("最低價", f"${kline_data['Low'].min():.2f}")
                with col3:
                    st.metric("收盤價", f"${kline_data['Close'].iloc[-1]:.2f}")
                with col4:
                    try:
                        change = ((kline_data["Close"].iloc[-1] - kline_data["Open"].iloc[0]) / kline_data["Open"].iloc[0]) * 100
                        st.metric("漲跌幅", f"{change:.2f}%")
                    except Exception:
                        st.metric("漲跌幅", "N/A")
            else:
                st.warning(f"⚠️ 無法取得 {stock_name} 的K線資料")

            st.markdown("---")
    else:
        st.info("👈 請選擇要分析的股票")

# ---- Tab 4 ----
with tab4:
    st.header("📋 詳細資料")

    st.subheader("完整數據表")
    full_display_df = summary_df[["股票代碼", "股票名稱", "外資買超", "內資買超", "自營商買超", "總買超", "連買天數", "評分"]].copy()
    full_display_df = full_display_df.sort_values("評分", ascending=False).reset_index(drop=True)
    full_display_df.index = full_display_df.index + 1

    display_full = full_display_df.copy()
    for c in ["外資買超", "內資買超", "自營商買超", "總買超"]:
        display_full[c] = display_full[c].apply(lambda x: f"{int(x):+d}")
    display_full["評分"] = display_full["評分"].apply(lambda x: f"{x:.1f}")

    st.dataframe(display_full, use_container_width=True)

    csv = full_display_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="📥 下載數據 (CSV)",
        data=csv,
        file_name=f"chip_monitor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    st.subheader("📊 統計摘要")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("監控股票數", len(summary_df))
    with col2:
        st.metric("買超股票數", len(summary_df[summary_df["總買超"] > 0]))
    with col3:
        st.metric("賣超股票數", len(summary_df[summary_df["總買超"] < 0]))
    with col4:
        st.metric("平均評分", f"{summary_df['評分'].mean():.1f}")
    with col5:
        st.metric("最高評分", f"{summary_df['評分'].max():.1f}")

    st.subheader("📈 評分統計分析")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=summary_df["評分"], nbinsx=15,
                                   marker_color="rgb(158, 202, 225)", name="股票數量"))
        fig.update_layout(title="全部股票評分分布", xaxis_title="評分", yaxis_title="股票數量", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        buy_count = len(summary_df[summary_df["總買超"] > 0])
        sell_count = len(summary_df[summary_df["總買超"] < 0])
        neutral_count = len(summary_df[summary_df["總買超"] == 0])

        fig = go.Figure(data=[go.Pie(
            labels=["買超", "賣超", "平手"],
            values=[buy_count, sell_count, neutral_count],
            marker=dict(colors=["#d32f2f", "#1976d2", "#90a4ae"])
        )])
        fig.update_layout(title="法人買賣超分布", height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 關於此應用\n- **數據源**: TWSE (T86)\n- **K線數據**: Yahoo Finance\n- **更新頻率**: 依開市日自動抓最近10日")

with col2:
    st.markdown("### 🎯 評分說明\n- **外資**: 30%\n- **內資(投信)**: 20%\n- **自營商**: 20%\n- **連買天數**: 30%")

with col3:
    st.markdown("### ⚠️ 免責聲明\n本平台僅供參考使用，不構成投資建議。股票投資存在風險，請理性投資。")

st.markdown(
    f"""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | v2.0 | 最近10個TWSE開市日
    </div>
    """,
    unsafe_allow_html=True
)
