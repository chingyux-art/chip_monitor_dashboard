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

st.title("📊 台股籌碼監控儀表板 v2.0（FinMind 版）")
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
    """
    把 FinMind 回傳的法人資料轉成你儀表板需要的 schema：
    日期、股票代碼、股票名稱、外資買超、內資買超、 自營商買超

    注意：FinMind 不同版本/資料集欄位命名可能不同，所以這裡做「多候選欄位」容錯。
    """
    if raw_df is None or len(raw_df) == 0:
        return pd.DataFrame()

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
