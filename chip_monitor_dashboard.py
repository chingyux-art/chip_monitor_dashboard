import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import re

# FinMind
from FinMind.data import DataLoader  # pip install FinMind  


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="台股籌碼監控儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

st.title("📊 台股籌碼監控儀表板 v2.1（FinMind 版）")
st.markdown("---")


# =========================
# Sidebar
# =========================
st.sidebar.header("⚙️ 設定")
data_source = st.sidebar.radio(
    "選擇數據源",
    options=["真實數據 (FinMind)", "模擬數據 (測試用)"],
    help="真實數據改用 FinMind API（包含三大法人買賣資料）"  # [1](https://ithelp.ithome.com.tw/articles/10341946)
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
# Helpers
# =========================
def to_int(x):
    if x is None:
        return 0
    s = str(x).strip()
    if s in ("", "--", "—", "NaN", "nan", "None"):
        return 0
    s = s.replace(",", "").replace(" ", "")
    s = s.replace("(", "-").replace(")", "")
    m = re.findall(r"-?\d+", s)
    return int(m[0]) if m else 0


def find_col(cols, candidates):
    """在欄位名稱中找第一個命中的欄位"""
    cols_lower = {str(c).lower(): c for c in cols}
    for key in candidates:
        k = key.lower()
        # 1) exact
        if k in cols_lower:
            return cols_lower[k]
        # 2) contains
        for cl, orig in cols_lower.items():
            if k in cl:
                return orig
    return None


@st.cache_resource
def get_finmind_loader():
    """
    初始化 FinMind DataLoader。
    - FinMind 支援 token，能提高 API request 上限（文件與套件說明都有提到 token 使用）
    """
    token = ""
    # Streamlit Cloud secrets 或本地 secrets.toml
    if hasattr(st, "secrets") and "FINMIND_TOKEN" in st.secrets:
        token = st.secrets["FINMIND_TOKEN"]

    if token:
        # 兩種寫法都常見：DataLoader(token=...) 或 login_by_token
        dl = DataLoader(token=token)
    else:
        dl = DataLoader()
    return dl

def normalize_finmind_institutional(raw_df, stock_code, stock_name):
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    def get(col_name):
        return df[col_name] if col_name in df.columns else 0

    # ✅ 外資
    foreign_buy = get("Foreign_Investor_Buy")
    foreign_sell = get("Foreign_Investor_Sell")
    foreign_net = foreign_buy - foreign_sell

    # ✅ 投信
    trust_buy = get("Investment_Trust_Buy")
    trust_sell = get("Investment_Trust_Sell")
    trust_net = trust_buy - trust_sell

    # ✅ 自營商（分兩種就加起來）
    dealer_buy = get("Dealer_Buy") + get("Dealer_Hedging_Buy")
    dealer_sell = get("Dealer_Sell") + get("Dealer_Hedging_Sell")
    dealer_net = dealer_buy - dealer_sell

    out = pd.DataFrame({
        "日期": df["date"],
        "股票代碼": stock_code,
        "股票名稱": stock_name,
        "外資買超": foreign_net.astype(int),
        "內資買超": trust_net.astype(int),
        "自營商買超": dealer_net.astype(int),
    })

    return out

    cols = list(raw_df.columns)

    # date & stock_id
    date_col = find_col(cols, ["date"])
    stock_id_col = find_col(cols, ["stock_id", "stockid", "stock"])
    if date_col is None:
        # 若找不到 date，就不處理
        return pd.DataFrame()

    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # 外資、投信、自營商 欄位候選（先找 net，再找 buy/sell 來算）
    # 這些關鍵字做成「包含匹配」，可涵蓋多數命名差異
    foreign_net = find_col(cols, ["foreign", "外資", "foreign_investor", "foreign investor", "foreign net", "foreign_buy_sell", "foreign_buy-sell", "Foreign_Investor_Buy_Sell"])
    trust_net = find_col(cols, ["trust", "投信", "investment trust", "institutional_trust", "trust net", "trust_buy_sell", "Investment_Trust_Buy_Sell"])
    dealer_net = find_col(cols, ["dealer", "自營商", "dealer net", "dealer_buy_sell", "Dealer_Buy_Sell", "dealer_total"])

    # 若欄位其實是 buy/sell，嘗試推導 net
    def compute_net(net_col_guess, buy_keys, sell_keys):
        if net_col_guess is not None and net_col_guess in df.columns:
            # 若看起來像 net 欄位，直接轉數字
            return df[net_col_guess].apply(to_int)

        buy_col = find_col(cols, buy_keys)
        sell_col = find_col(cols, sell_keys)
        if buy_col is not None and sell_col is not None:
            return df[buy_col].apply(to_int) - df[sell_col].apply(to_int)

        # 找不到就回 0
        return pd.Series([0] * len(df))

    foreign_series = compute_net(
        foreign_net,
        buy_keys=["foreign buy", "外資買進", "foreign_investor_buy", "foreign_buy"],
        sell_keys=["foreign sell", "外資賣出", "foreign_investor_sell", "foreign_sell"]
    )
    trust_series = compute_net(
        trust_net,
        buy_keys=["trust buy", "投信買進", "investment_trust_buy", "trust_buy"],
        sell_keys=["trust sell", "投信賣出", "investment_trust_sell", "trust_sell"]
    )
    dealer_series = compute_net(
        dealer_net,
        buy_keys=["dealer buy", "自營商買進", "dealer_buy"],
        sell_keys=["dealer sell", "自營商賣出", "dealer_sell"]
    )

    out = pd.DataFrame({
        "日期": df[date_col],
        "股票代碼": stock_code,
        "股票名稱": stock_name,
        "外資買超": foreign_series.astype(int),
        "內資買超": trust_series.astype(int),      # 你原本「內資」概念最常見就是投信 
        "自營商買超": dealer_series.astype(int),
    })

    return out


@st.cache_data(ttl=3600)
def finmind_fetch_institutional(stock_code, stock_name, start_date, end_date):
    """
    用 FinMind 取得單一股票在區間內的法人資料。
    iT 邦幫忙示例顯示可用 DataLoader.taiwan_stock_institutional_investors(...) 取資料。
    """
    dl = get_finmind_loader()
    # FinMind 常用接口：taiwan_stock_institutional_investors
    raw = dl.taiwan_stock_institutional_investors(
        stock_id=stock_code,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d")
    )
    return normalize_finmind_institutional(raw, stock_code, stock_name)


def get_last_n_trading_days_from_finmind(n=10, probe_stock="2330", lookback_days=90):
    """
    用 FinMind 的資料「反推最近 n 個真實交易日」：
    - 先抓一檔流動性高的股票（預設 2330）最近 lookback_days 的法人資料
    - 取其日期去重排序，取最後 n 個
    這樣比「排除週末」更接近真實開市日（會自動排除休市日）。
    """
    end = datetime.now().date()
    start = end - timedelta(days=lookback_days)

    df_probe = finmind_fetch_institutional(probe_stock, CODE_TO_NAME.get(probe_stock, probe_stock), start, end)
    if df_probe.empty:
        return []

    days = sorted(df_probe["日期"].dt.date.unique())
    if len(days) >= n:
        return days[-n:]
    return days


@st.cache_data(ttl=3600)
def load_industry_data():
    """加載產業分類 CSV 資料"""
    try:
        df = pd.read_csv("台股產業概念股完整分類_含詳細說明.csv", encoding="utf-8-sig")
        return df
    except Exception as e:
        st.warning(f"⚠️ 無法加載產業分類資料：{e}")
        return pd.DataFrame()


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
# Main: get last 10 trading days
# =========================
st.info("🚀 初始化中：將以 FinMind 資料推算最近 10 個實際交易日…")  # [1](https://ithelp.ithome.com.tw/articles/10341946)

with st.spinner("正在取得最近 10 個交易日（FinMind）…"):
    open_days = get_last_n_trading_days_from_finmind(n=10, probe_stock="2330", lookback_days=120)

if not open_days:
    st.warning("⚠️ 無法從 FinMind 取得交易日資料，將改用模擬數據流程。")
    data_source = "模擬數據 (測試用)"
    # fallback：近 10 個平日
    open_days = []
    d = datetime.now().date()
    while len(open_days) < 10:
        if d.weekday() < 5:
            open_days.append(d)
        d -= timedelta(days=1)
    open_days = sorted(open_days)

start_date, end_date = open_days[0], open_days[-1]
st.success(f"✅ 分析區間：最近 10 個交易日（{start_date} ～ {end_date}）")


# =========================
# Load data
# =========================
if data_source == "真實數據 (FinMind)":
    st.warning("📡 正在從 FinMind 取得 20 檔股票之三大法人資料…")  # [1](https://ithelp.ithome.com.tw/articles/10341946)
    with st.spinner("正在加載數據..."):
        all_data = []
        # 為了確保涵蓋 10 個交易日，往前多抓一些緩衝天數
        fetch_start = (pd.to_datetime(open_days[0]) - pd.Timedelta(days=14)).date()
        fetch_end = pd.to_datetime(open_days[-1]).date()

        for stock_code, stock_name in STOCKS_LIST:
            try:
                df_one = finmind_fetch_institutional(stock_code, stock_name, fetch_start, fetch_end)
                if not df_one.empty:
                    # 只保留最近 10 個交易日
                    df_one["日期_純"] = df_one["日期"].dt.date
                    df_one = df_one[df_one["日期_純"].isin(open_days)].drop(columns=["日期_純"])
                    all_data.append(df_one)
            except Exception:
                pass

        if all_data:
            institutional_df = pd.concat(all_data, ignore_index=True)
        else:
            st.warning("⚠️ FinMind 真實數據抓取失敗，自動切換到模擬數據")
            institutional_df = generate_institutional_data(open_days, STOCKS_LIST)
else:
    institutional_df = generate_institutional_data(open_days, STOCKS_LIST)

# 若仍為空，直接停止
if institutional_df.empty:
    st.error("⚠️ 無法取得任何資料（真實/模擬皆失敗），請檢查環境與套件。")
    st.stop()

# 添加總買超
institutional_df["總買超"] = (
    institutional_df["外資買超"] +
    institutional_df["內資買超"] +
    institutional_df["自營商買超"]
)

# 計算連買天數
consecutive_buying_df = calculate_buy_streaks(institutional_df)

# 匯總最新數據（每檔股票取最後一筆）
latest_data = []
for stock_code, stock_name in STOCKS_LIST:
    stock_data = institutional_df[institutional_df["股票代碼"] == stock_code].sort_values("日期")
    if len(stock_data) == 0:
        # 若該股真實資料缺漏，用 0 填（避免整個 dashboard 爆）
        latest_data.append({
            "股票代碼": stock_code,
            "股票名稱": stock_name,
            "外資買超": 0,
            "內資買超": 0,
            "自營商買超": 0,
            "總買超": 0,
            "連買天數": 0,
            "評分": calculate_score({
                "外資買超": 0, "內資買超": 0, "自營商買超": 0, "連買天數": 0
            })
        })
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
if summary_df.empty:
    st.error("⚠️ summary_df 為空，無法顯示圖表。")
    st.stop()


# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["法人籌碼", "熱門股票", "K線分析", "詳細資料", "產業分類"])

# ---- Tab 1: 法人籌碼 ----
with tab1:
    st.header("📈 法人籌碼分析")
    
    # 新增產業篩選
    industry_df = load_industry_data()
    if not industry_df.empty:
        unique_industries = sorted(industry_df["產業分類"].unique())
        selected_industry = st.selectbox(
            "🏭 選擇產業分類（選擇後只顯示該產業個股）",
            options=["全部"] + unique_industries,
            key="tab1_industry"
        )
        
        # 根據選擇篩選股票
        if selected_industry != "全部":
            industry_stocks = industry_df[industry_df["產業分類"] == selected_industry]["個股代碼名稱"].tolist()
            # 提取股票代碼（格式為 "代碼 (名稱)"）
            stock_codes = [code.split()[0] for code in industry_stocks if code]
            display_df = summary_df[summary_df["股票代碼"].isin(stock_codes)].copy()
            st.info(f"🏭 已篩選產業：**{selected_industry}** - 共 {len(display_df)} 檔個股")
        else:
            display_df = summary_df.copy()
    else:
        display_df = summary_df.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_foreign = display_df["外資買超"].mean()
        st.metric("平均外資買超", f"{avg_foreign:.0f}")
    with col2:
        avg_domestic = display_df["內資買超"].mean()
        st.metric("平均內資買超(投信)", f"{avg_domestic:.0f}")
    with col3:
        avg_dealer = display_df["自營商買超"].mean()
        st.metric("平均自營商買超", f"{avg_dealer:.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(data=[
            go.Bar(x=display_df["股票名稱"], y=display_df["外資買超"], name="外資"),
            go.Bar(x=display_df["股票名稱"], y=display_df["內資買超"], name="內資(投信)"),
            go.Bar(x=display_df["股票名稱"], y=display_df["自營商買超"], name="自營商"),
        ])
        fig.update_layout(title="各法人買賣超對比", barmode="group", height=400, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 只顯示篩選後的連買天數 Top 10
        filtered_consecutive = consecutive_buying_df[consecutive_buying_df["股票代碼"].isin(display_df["股票代碼"])]
        if len(filtered_consecutive) > 0:
            top_consecutive = filtered_consecutive.nlargest(10, "連買天數")
            fig = px.bar(top_consecutive, x="股票名稱", y="連買天數", color="連買天數",
                         title="連買天數 Top 10", color_continuous_scale="Viridis")
            fig.update_layout(height=400, hovermode="x")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("最新法人籌碼數據")
    display_show = display_df[["股票代碼", "股票名稱", "外資買超", "內資買超", "自營商買超", "總買超"]].copy()
    for c in ["外資買超", "內資買超", "自營商買超", "總買超"]:
        display_show[c] = display_show[c].apply(lambda x: f"{int(x):+d}")
    st.dataframe(display_show, use_container_width=True, hide_index=True)


# ---- Tab 2: 熱門股票 ----
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
        fig.add_trace(go.Histogram(x=top_20["評分"], nbinsx=15, marker_color="rgb(55, 83, 109)", opacity=0.7))
        fig.update_layout(title="評分分布", xaxis_title="評分 (0-100)", yaxis_title="股票數量", height=400, showlegend=False)
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


# ---- Tab 3: K線分析 ----
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
                fig.update_layout(title=f"{stock_name} K線圖", yaxis_title="股價 (TWD)", xaxis_title="日期",
                                  template="plotly_white", height=500, hovermode="x unified")
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


# ---- Tab 4: 詳細資料 ----
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
        fig.add_trace(go.Histogram(x=summary_df["評分"], nbinsx=15, marker_color="rgb(158, 202, 225)"))
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


# ---- Tab 5: 產業分類 ----
with tab5:
    st.header("🏭 台股產業概念股完整分類")
    
    industry_df = load_industry_data()
    
    if not industry_df.empty:
        # 顯示所有產業分類
        unique_industries = sorted(industry_df["產業分類"].unique())
        
        st.subheader("📌 選擇產業分類")
        selected_category = st.selectbox(
            "請選擇要查看的產業",
            options=unique_industries,
            key="tab5_industry"
        )
        
        # 篩選該產業的所有個股
        category_df = industry_df[industry_df["產業分類"] == selected_category].copy()
        
        st.markdown(f"### {selected_category}")
        st.info(f"共有 **{len(category_df)}** 檔個股在此產業分類")
        
        # 顯示該產業的詳細資訊
        if len(category_df) > 0:
            # 準備顯示用的資料
            display_cols = ["序號", "個股代碼名稱", "產業定位", "公司業務說明", "在該產業的地位與優勢"]
            available_cols = [col for col in display_cols if col in category_df.columns]
            
            display_table = category_df[available_cols].copy()
            
            # 使用表格顯示
            st.dataframe(display_table, use_container_width=True, hide_index=True)
            
            # 提取股票代碼，展示該產業在籌碼監控中的表現
            st.markdown("---")
            st.subheader(f"📊 {selected_category} 籌碼表現")
            
            # 提取股票代碼（格式為 "代碼 (名稱)"）
            stock_codes = []
            for code_name in category_df["個股代碼名稱"].tolist():
                if pd.notna(code_name):
                    code = str(code_name).split()[0]
                    stock_codes.append(code)
            
            # 在籌碼監控資料中篩選該產業的個股
            industry_performance = summary_df[summary_df["股票代碼"].isin(stock_codes)].copy()
            
            if len(industry_performance) > 0:
                # 按評分排序
                industry_performance = industry_performance.sort_values("評分", ascending=False)
                
                # 顯示該產業的籌碼排名
                st.subheader("該產業在籌碼監控中的排名")
                perf_display = industry_performance[["股票代碼", "股票名稱", "外資買超", "內資買超", "自營商買超", "總買超", "連買天數", "評分"]].copy()
                
                for c in ["外資買超", "內資買超", "自營商買超", "總買超"]:
                    perf_display[c] = perf_display[c].apply(lambda x: f"{int(x):+d}")
                perf_display["評分"] = perf_display["評分"].apply(lambda x: f"{x:.1f}")
                
                st.dataframe(perf_display, use_container_width=True, hide_index=True)
                
                # 統計資訊
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("監控中的個股數", len(industry_performance))
                with col2:
                    st.metric("平均評分", f"{industry_performance['評分'].mean():.1f}")
                with col3:
                    st.metric("買超個股數", len(industry_performance[industry_performance["總買超"] > 0]))
                with col4:
                    st.metric("最高評分", f"{industry_performance['評分'].max():.1f}")
            else:
                st.info(f"ℹ️ 該產業個股目前未納入籌碼監控清單中")
    else:
        st.warning("⚠️ 無法加載產業分類資料")


st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 關於此應用\n- **數據源**: FinMind（三大法人買賣資料）\n- **K線數據**: Yahoo Finance\n- **區間**: 最近 10 個交易日")  # [1](https://ithelp.ithome.[...]

with col2:
    st.markdown("### 🎯 評分說明\n- **外資**: 30%\n- **內資(投信)**: 20%\n- **自營商**: 20%\n- **連買天數**: 30%")

with col3:
    st.markdown("### ⚠️ 免責聲明\n本平台僅供參考使用，不構成投資建議。股票投資存在風險，請理性投資。")

st.markdown(
    f"""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | v2.1 | FinMind Data
    </div>
    """,
    unsafe_allow_html=True
)
