import html
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from FinMind.data import DataLoader

st.set_page_config(page_title="台股族群監控", page_icon="📊", layout="wide")
st.title("📊 台股族群監控 App")

CSV_PATH = Path("Group.csv")


CARD_CSS = """
<style>
.stock-card-title {font-size: 1.05rem; font-weight: 800; margin-bottom: 0.15rem;}
.stock-card-subtitle {color: #6b7280; font-size: 0.82rem; margin-bottom: 0.75rem;}
.metric-row {margin: 0.55rem 0 0.7rem;}
.metric-label {display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.78rem; color: #374151;}
.metric-value {font-weight: 700; white-space: nowrap;}
.metric-track {height: 0.55rem; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin-top: 0.25rem;}
.metric-fill {height: 100%; border-radius: 999px; min-width: 0.25rem;}
.metric-fill.up {background: linear-gradient(90deg, #fca5a5, #dc2626);}
.metric-fill.down {background: linear-gradient(90deg, #86efac, #16a34a);}
.metric-fill.neutral {background: linear-gradient(90deg, #cbd5e1, #64748b);}
.metric-note {font-size: 0.72rem; color: #6b7280; margin-top: 0.15rem;}
</style>
"""


def _parse_stock_code(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).strip()
    m = re.search(r"\d{4,6}", t)
    return m.group(0) if m else ""


def _parse_stock_name(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).strip()
    m = re.match(r"([^\d\(\)\s]+)", t)
    return m.group(1) if m else t


@st.cache_data(ttl=5)
def load_group_csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def ensure_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(index=df.index)

    def pick_col(candidates):
        matched = [c for c in df.columns if any(k in str(c) for k in candidates)]
        if not matched:
            return pd.Series([""] * len(df), index=df.index)
        if len(matched) == 1:
            return df[matched[0]].fillna("")
        subset = df[matched].copy().replace({np.nan: ""}).astype(str)
        return subset.apply(lambda r: next((v for v in r if str(v).strip()), ""), axis=1)

    normalized["族群"] = pick_col(["產業分類", "產業", "族群"])
    normalized["個股代碼名稱"] = pick_col(["代碼名稱", "個股代碼", "股票代碼", "ticker"])
    normalized["產業定位"] = pick_col(["產業定位", "定位"])
    normalized["公司業務說明"] = pick_col(["公司業務說明", "業務", "說明", "介紹"])
    normalized["在該產業的地位與優勢"] = pick_col(["在該產業的地位與優勢", "地位", "優勢"])
    return normalized


@st.cache_data(ttl=900)
def fetch_ohlcv(stock_code: str, period: str = "6mo") -> pd.DataFrame:
    for suffix in [".TW", ".TWO"]:
        try:
            data = yf.download(f"{stock_code}{suffix}", period=period, interval="1d", progress=False, auto_adjust=False)
            if data is not None and len(data) > 1:
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                return data.dropna()
        except Exception:
            continue
    return pd.DataFrame()


def latest_price_change(stock_code: str):
    df = fetch_ohlcv(stock_code, period="1mo")
    if df.empty or len(df) < 2:
        return np.nan, np.nan, np.nan
    c1, c0 = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
    pct = (c1 - c0) / c0 * 100 if c0 else np.nan
    return c1, c0, pct


def _safe_pct(new_value: float, old_value: float) -> float:
    if pd.isna(new_value) or pd.isna(old_value) or old_value == 0:
        return np.nan
    return (new_value - old_value) / old_value * 100


def _format_number(value: float, digits: int = 2, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}{suffix}"


def _bar_html(label: str, value: float, note: str, positive_base: float = 0.0, max_abs: float = 10.0, unit=None) -> str:
    if pd.isna(value):
        css_class = "neutral"
        width = 8
        shown = "N/A"
    else:
        delta = value - positive_base
        css_class = "up" if delta >= 0 else "down"
        width = min(max(abs(delta) / max_abs * 100, 8), 100)
        shown_unit = unit if unit is not None else ("%" if positive_base == 0 else "x")
        shown = _format_number(value, 2, shown_unit)
    return f"""
    <div class="metric-row">
      <div class="metric-label"><span>{label}</span><span class="metric-value">{shown}</span></div>
      <div class="metric-track"><div class="metric-fill {css_class}" style="width:{width:.1f}%"></div></div>
      <div class="metric-note">{note}</div>
    </div>
    """


def stock_activity_metrics(stock_code: str) -> dict:
    df = fetch_ohlcv(stock_code, period="3mo")
    empty_metrics = {
        "latest_close": np.nan,
        "previous_close": np.nan,
        "daily_pct": np.nan,
        "avg_5d_pct": np.nan,
        "cumulative_5d_pct": np.nan,
        "latest_volume": np.nan,
        "previous_volume": np.nan,
        "volume_day_ratio": np.nan,
        "volume_5d_vs_20d": np.nan,
        "last_date": "",
    }
    if df.empty or len(df) < 2:
        return empty_metrics

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    daily_return = close.pct_change() * 100
    latest_close = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    latest_volume = float(volume.iloc[-1])
    previous_volume = float(volume.iloc[-2])
    volume_20d_avg = volume.tail(20).mean() if len(volume) >= 20 else volume.mean()
    last_index = df.index[-1]

    metrics = empty_metrics.copy()
    metrics.update({
        "latest_close": latest_close,
        "previous_close": previous_close,
        "daily_pct": _safe_pct(latest_close, previous_close),
        "avg_5d_pct": float(daily_return.tail(5).mean()) if len(daily_return.dropna()) else np.nan,
        "cumulative_5d_pct": _safe_pct(latest_close, float(close.iloc[-6])) if len(close) >= 6 else np.nan,
        "latest_volume": latest_volume,
        "previous_volume": previous_volume,
        "volume_day_ratio": latest_volume / previous_volume if previous_volume else np.nan,
        "volume_5d_vs_20d": volume.tail(5).mean() / volume_20d_avg if volume_20d_avg else np.nan,
        "last_date": last_index.strftime("%Y-%m-%d") if hasattr(last_index, "strftime") else str(last_index),
    })
    return metrics


def stock_card_html(stock: dict) -> str:
    metrics = stock["metrics"]
    title = html.escape(str(stock["名稱"] or stock["個股代碼名稱"] or stock["代碼"]))
    subtitle = html.escape(f"{stock['代碼']}｜{stock['產業定位'] or '未填產業定位'}")
    price_note = (
        html.escape(f"最新 {_format_number(metrics['latest_close'])} / 前日 {_format_number(metrics['previous_close'])}")
        if not pd.isna(metrics["latest_close"])
        else "尚無股價資料"
    )
    volume_note = (
        html.escape(f"最新量 {_format_number(metrics['latest_volume'], 0)} / 前日量 {_format_number(metrics['previous_volume'], 0)}")
        if not pd.isna(metrics["latest_volume"])
        else "尚無成交量資料"
    )
    return f"""
    <div class="stock-card-title">{title}</div>
    <div class="stock-card-subtitle">{subtitle}｜資料日 {metrics['last_date'] or 'N/A'}</div>
    {_bar_html('最新一日股價漲跌幅', metrics['daily_pct'], price_note, 0, 10)}
    {_bar_html('近5日平均漲幅', metrics['avg_5d_pct'], '最近5個交易日每日漲跌幅平均', 0, 10)}
    {_bar_html('近5日累積漲幅', metrics['cumulative_5d_pct'], '最新收盤相對5個交易日前收盤', 0, 20)}
    {_bar_html('最新一日成交量 / 前一日', metrics['volume_day_ratio'], volume_note, 1, 1.5)}
    {_bar_html('近5日成交量 / 近20日均量', metrics['volume_5d_vs_20d'], '以近5日均量除以近20日均量', 1, 1.5)}
    """


@st.dialog("個股詳細說明")
def show_stock_detail(stock: dict):
    metrics = stock["metrics"]
    title = stock["名稱"] or stock["個股代碼名稱"] or stock["代碼"]
    st.subheader(f"{title} ({stock['代碼']})")
    st.caption(f"族群：{stock['族群']}｜最新資料日：{metrics['last_date'] or 'N/A'}")
    c1, c2, c3 = st.columns(3)
    c1.metric("最新收盤", _format_number(metrics["latest_close"]), _format_number(metrics["daily_pct"], 2, "%"))
    c2.metric("近5日平均漲幅", _format_number(metrics["avg_5d_pct"], 2, "%"))
    c3.metric("量比（近5日/20日）", _format_number(metrics["volume_5d_vs_20d"], 2, "x"))
    st.markdown("#### 公司與產業說明")
    st.write(f"**產業定位：** {stock['產業定位'] or '未提供'}")
    st.write(f"**公司業務說明：** {stock['公司業務說明'] or '未提供'}")
    st.write(f"**在該產業的地位與優勢：** {stock['在該產業的地位與優勢'] or '未提供'}")

    kdf = fetch_ohlcv(stock["代碼"], period="1y")
    if not kdf.empty:
        fig = go.Figure(data=[go.Candlestick(x=kdf.index, open=kdf["Open"], high=kdf["High"], low=kdf["Low"], close=kdf["Close"])])
        fig.update_layout(height=360, xaxis_rangeslider_visible=False, title=f"{title} 近一年K線")
        st.plotly_chart(fig, width="stretch")


@st.dialog("近一年K線")
def show_kline_dialog(stock_name: str, stock_code: str):
    st.subheader(f"{stock_name} ({stock_code})")
    kdf = fetch_ohlcv(stock_code, period="1y")
    if kdf.empty:
        st.info("目前無法取得此個股的K線資料。")
        return
    fig = go.Figure(data=[go.Candlestick(x=kdf.index, open=kdf["Open"], high=kdf["High"], low=kdf["Low"], close=kdf["Close"])])
    fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{stock_name} 近一年K線")
    st.plotly_chart(fig, width="stretch")


def score_card_html(stock: dict) -> str:
    title = html.escape(str(stock["個股名稱"] or stock["代碼"]))
    code = html.escape(str(stock["代碼"]))
    score = stock.get("分數", np.nan)
    items = stock.get("細項") or []
    item_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in items[:4]) or "<li>尚無符合加分條件</li>"
    return f"""
    <div class="stock-card-title">{title}</div>
    <div class="stock-card-subtitle">{code}｜技術面評分</div>
    {_bar_html('技術面分數', score, '依 MACD、均線、量能與近月趨勢累計', 0, 6, '分')}
    <ul class="metric-note">{item_html}</ul>
    """


@st.cache_data(ttl=900)
def fetch_financial_detail(stock_code: str) -> dict:
    detail = {"fast_info": {}, "quarterly_financials": pd.DataFrame(), "major_holders": pd.DataFrame()}
    for suffix in [".TW", ".TWO"]:
        try:
            ticker = yf.Ticker(f"{stock_code}{suffix}")
            detail["fast_info"] = dict(ticker.fast_info or {})
            qf = ticker.quarterly_financials
            if isinstance(qf, pd.DataFrame):
                detail["quarterly_financials"] = qf
            holders = ticker.major_holders
            if isinstance(holders, pd.DataFrame):
                detail["major_holders"] = holders
            return detail
        except Exception:
            continue
    return detail


def financial_card_html(stock: dict) -> str:
    title = html.escape(str(stock["個股名稱"] or stock["代碼"]))
    code = html.escape(str(stock["代碼"]))
    score = stock.get("分數", np.nan)
    items = stock.get("細項") or []
    item_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in items[:4]) or "<li>尚無符合加分條件</li>"
    return f"""
    <div class="stock-card-title">{title}</div>
    <div class="stock-card-subtitle">{code}｜基本面評分</div>
    {_bar_html('基本面分數', score, '依市值、季營收與半年均值等資料累計', 0, 4, '分')}
    <ul class="metric-note">{item_html}</ul>
    """


@st.dialog("基本面詳細資料")
def show_financial_detail(stock: dict):
    st.subheader(f"{stock['個股名稱']} ({stock['代碼']})")
    st.caption(f"基本面分數：{_format_number(stock.get('分數', np.nan), 1, '分')}")
    st.markdown("#### 評分細項")
    details = stock.get("細項") or ["尚無符合加分條件"]
    for item in details:
        st.write(f"- {item}")

    source_row = stock.get("source_row", {})
    if source_row:
        st.markdown("#### 公司與產業說明")
        st.write(f"**產業定位：** {source_row.get('產業定位') or '未提供'}")
        st.write(f"**公司業務說明：** {source_row.get('公司業務說明') or '未提供'}")
        st.write(f"**在該產業的地位與優勢：** {source_row.get('在該產業的地位與優勢') or '未提供'}")

    detail = fetch_financial_detail(stock["代碼"])
    fast_info = detail.get("fast_info", {})
    if fast_info:
        st.markdown("#### Yahoo Finance 快速資訊")
        cols = st.columns(3)
        cols[0].metric("市值", _format_number(fast_info.get("market_cap"), 0))
        cols[1].metric("最近價格", _format_number(fast_info.get("last_price")))
        cols[2].metric("年初至今最高", _format_number(fast_info.get("year_high")))

    qf = detail.get("quarterly_financials", pd.DataFrame())
    if isinstance(qf, pd.DataFrame) and not qf.empty:
        st.markdown("#### 季度財務資料")
        st.dataframe(qf, width="stretch")


def chip_card_html(stock: dict) -> str:
    title = html.escape(str(stock.get("個股", stock.get("代碼", ""))))
    return f"""
    <div class="stock-card-title">{title}</div>
    <div class="stock-card-subtitle">買超單位：張｜紅色買超、綠色賣超</div>
    {_bar_html('外資 5日買超', stock.get('外資_5日', np.nan), '最近5個交易日合計', 0, 10000, '張')}
    {_bar_html('外資 10日買超', stock.get('外資_10日', np.nan), '最近10個交易日合計', 0, 20000, '張')}
    {_bar_html('外資 20日買超', stock.get('外資_20日', np.nan), '最近20個交易日合計', 0, 40000, '張')}
    {_bar_html('投信 5日買超', stock.get('投信_5日', np.nan), '最近5個交易日合計', 0, 3000, '張')}
    {_bar_html('投信 10日買超', stock.get('投信_10日', np.nan), '最近10個交易日合計', 0, 6000, '張')}
    {_bar_html('投信 20日買超', stock.get('投信_20日', np.nan), '最近20個交易日合計', 0, 12000, '張')}
    {_bar_html('自營商 5日買超', stock.get('自營商_5日', np.nan), '最近5個交易日合計', 0, 3000, '張')}
    {_bar_html('自營商 10日買超', stock.get('自營商_10日', np.nan), '最近10個交易日合計', 0, 6000, '張')}
    {_bar_html('自營商 20日買超', stock.get('自營商_20日', np.nan), '最近20個交易日合計', 0, 12000, '張')}
    """


def render_card_grid(cards, html_func, cols_per_row: int, button_label=None, button_key_prefix: str = "card", on_click=None):
    for start in range(0, len(cards), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, stock in zip(cols, cards[start:start + cols_per_row]):
            with col:
                with st.container(border=True):
                    st.markdown(html_func(stock), unsafe_allow_html=True)
                    if button_label and on_click:
                        if st.button(button_label, key=f"{button_key_prefix}_{stock['代碼']}", width="stretch"):
                            on_click(stock)


def show_tech_score_explanation():
    with st.expander("技術面評分說明", expanded=True):
        st.markdown(
            """
            **最高 6 分。** 分數越高代表近期技術條件越多同時成立，僅作為排序與觀察輔助。

            | 條件 | 加分 |
            |---|---:|
            | MACD 由下往上穿越訊號線（黃金交叉） | +2 |
            | 最新收盤價站上 20 日均線 | +1 |
            | 最新收盤價站上 60 日均線 | +1 |
            | 最新成交量大於前 4 個交易日均量的 1.5 倍 | +1 |
            | 最新收盤價高於約 1 個月前（22 個交易日前） | +1 |
            """
        )


def show_financial_score_explanation():
    with st.expander("基本面評分說明", expanded=True):
        st.markdown(
            """
            **最高 4 分。** 分數越高代表目前可取得的基本面與股價相對強度條件越多同時成立。

            | 條件 | 加分 |
            |---|---:|
            | Yahoo Finance 可取得有效市值資料 | +1 |
            | 最新季度營收高於前一季營收 | +2 |
            | 最新收盤價高於近半年均值（120 日均線，至少 20 筆資料） | +1 |
            """
        )


def tech_score(stock_code: str) -> dict:
    df = fetch_ohlcv(stock_code, period="1y")
    base = {"代碼": stock_code, "分數": 0.0, "細項": []}
    if df.empty:
        return base
    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    if len(macd) >= 2 and macd.iloc[-2] < signal.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        base["分數"] += 2
        base["細項"].append("MACD 黃金交叉 +2")
    if close.iloc[-1] > close.rolling(20).mean().iloc[-1]:
        base["分數"] += 1
        base["細項"].append("站上20日均線 +1")
    if close.iloc[-1] > close.rolling(60).mean().iloc[-1]:
        base["分數"] += 1
        base["細項"].append("站上60日均線 +1")
    if len(vol) >= 5 and vol.iloc[-1] > vol.iloc[-5:-1].mean() * 1.5:
        base["分數"] += 1
        base["細項"].append("量能放大 +1")
    if len(close) >= 22 and close.iloc[-1] > close.iloc[-22]:
        base["分數"] += 1
        base["細項"].append("近一月趨勢向上 +1")
    return base


def fetch_financial_score(stock_code: str) -> dict:
    out = {"代碼": stock_code, "分數": 0.0, "細項": []}
    for suffix in [".TW", ".TWO"]:
        try:
            t = yf.Ticker(f"{stock_code}{suffix}")
            info = t.fast_info or {}
            if info.get("market_cap") and info["market_cap"] > 0:
                out["分數"] += 1
                out["細項"].append("市值資料有效 +1")

            qf = t.quarterly_financials
            if isinstance(qf, pd.DataFrame) and not qf.empty:
                rev_key = next((k for k in qf.index if "Revenue" in str(k)), None)
                if rev_key:
                    rev = qf.loc[rev_key].dropna().astype(float)
                    if len(rev) >= 2 and rev.iloc[0] > rev.iloc[1]:
                        out["分數"] += 2
                        out["細項"].append("最新季營收成長 +2")
            close = fetch_ohlcv(stock_code, period="6mo")
            if not close.empty:
                c = close["Close"].astype(float)
                if c.iloc[-1] > c.rolling(120, min_periods=20).mean().iloc[-1]:
                    out["分數"] += 1
                    out["細項"].append("股價強於半年均值 +1")
            return out
        except Exception:
            continue
    return out


def _institution_bucket(name: str) -> str:
    text = str(name)
    if "投信" in text or "Investment_Trust" in text:
        return "投信"
    if "自營" in text or "Dealer" in text:
        return "自營商"
    if "外資" in text or "Foreign" in text:
        return "外資"
    return "其他"


@st.cache_data(ttl=1800)
def fetch_institutional_summary(stock_code: str, end_date: str) -> dict:
    start_date = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=120)).strftime("%Y-%m-%d")
    base = {
        "代碼": stock_code,
        "外資_5日": np.nan,
        "外資_10日": np.nan,
        "外資_20日": np.nan,
        "投信_5日": np.nan,
        "投信_10日": np.nan,
        "投信_20日": np.nan,
        "自營商_5日": np.nan,
        "自營商_10日": np.nan,
        "自營商_20日": np.nan,
    }
    try:
        api = DataLoader()
        df = api.taiwan_stock_institutional_investors(stock_id=stock_code, start_date=start_date, end_date=end_date)
    except Exception:
        return base

    if df is None or df.empty:
        return base

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["bucket"] = df["name"].apply(_institution_bucket)
    df["net"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0) - pd.to_numeric(df["sell"], errors="coerce").fillna(0)
    grouped = df[df["bucket"].isin(["外資", "投信", "自營商"])].groupby(["date", "bucket"], as_index=False)["net"].sum()
    trade_dates = sorted(grouped["date"].unique())
    if not trade_dates:
        return base

    for bucket in ["外資", "投信", "自營商"]:
        bucket_df = grouped[grouped["bucket"] == bucket]
        for days in [5, 10, 20]:
            recent_dates = trade_dates[-days:]
            total_shares = bucket_df[bucket_df["date"].isin(recent_dates)]["net"].sum()
            base[f"{bucket}_{days}日"] = total_shares / 1000
    return base


if not CSV_PATH.exists():
    st.error("找不到 Group.csv，請確認檔案位於專案根目錄 chingyux-art/chip_monitor_dashboard/Group.csv。")
    st.stop()

raw_df = load_group_csv(str(CSV_PATH), CSV_PATH.stat().st_mtime)
group_df = ensure_group_columns(raw_df)
all_groups = sorted([g for g in group_df["族群"].dropna().unique() if str(g).strip()])

if not all_groups:
    st.error("CSV 缺少有效的『族群』資料")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["分頁一：族群與股價", "分頁二：技術面評分", "分頁三：基本面評分", "分頁四：籌碼統計"])

with tab1:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    g = st.selectbox("選擇族群", all_groups)
    subset = group_df[group_df["族群"] == g].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    subset["名稱"] = subset["個股代碼名稱"].apply(_parse_stock_name)
    cards = []
    with st.spinner("讀取股價與成交量資料中..."):
        for _, r in subset.iterrows():
            code = r["代碼"]
            if not code:
                continue
            cards.append({
                "族群": g,
                "代碼": code,
                "名稱": r["名稱"],
                "個股代碼名稱": r["個股代碼名稱"],
                "產業定位": r["產業定位"],
                "公司業務說明": r["公司業務說明"],
                "在該產業的地位與優勢": r["在該產業的地位與優勢"],
                "metrics": stock_activity_metrics(code),
            })

    if cards:
        cols_per_row = st.slider("每列顯示方框數", min_value=2, max_value=4, value=3)
        for start in range(0, len(cards), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, stock in zip(cols, cards[start:start + cols_per_row]):
                with col:
                    with st.container(border=True):
                        st.markdown(stock_card_html(stock), unsafe_allow_html=True)
                        if st.button("查看詳細說明", key=f"detail_{stock['代碼']}", width="stretch"):
                            show_stock_detail(stock)
    else:
        st.info("此族群沒有可用股票代碼")


with tab2:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    show_tech_score_explanation()
    g2 = st.selectbox("選擇族群（技術面）", all_groups)
    subset = group_df[group_df["族群"] == g2].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    subset["名稱"] = subset["個股代碼名稱"].apply(_parse_stock_name)
    name_map = dict(zip(subset["代碼"], subset["名稱"]))
    results = []
    with st.spinner("計算技術面評分中..."):
        for code in subset["代碼"].dropna().unique():
            if code:
                score = tech_score(code)
                score["個股名稱"] = name_map.get(code, code)
                results.append(score)
    if results:
        cards = pd.DataFrame(results).sort_values("分數", ascending=False).to_dict("records")
        cols_per_row = st.slider("每列顯示技術面方框數", min_value=2, max_value=4, value=3)
        render_card_grid(
            cards,
            score_card_html,
            cols_per_row,
            button_label="顯示K線",
            button_key_prefix="kline",
            on_click=lambda stock: show_kline_dialog(stock["個股名稱"], stock["代碼"]),
        )
    else:
        st.info("此族群沒有可用股票代碼")

with tab3:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    show_financial_score_explanation()
    g3 = st.selectbox("選擇族群（基本面）", all_groups)
    subset = group_df[group_df["族群"] == g3].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    subset["名稱"] = subset["個股代碼名稱"].apply(_parse_stock_name)
    name_map = dict(zip(subset["代碼"], subset["名稱"]))
    source_map = {row["代碼"]: row.to_dict() for _, row in subset.iterrows()}
    results = []
    with st.spinner("計算基本面評分中..."):
        for code in subset["代碼"].dropna().unique():
            if code:
                score = fetch_financial_score(code)
                score["個股名稱"] = name_map.get(code, code)
                score["source_row"] = source_map.get(code, {})
                results.append(score)
    if results:
        cards = pd.DataFrame(results).sort_values("分數", ascending=False).to_dict("records")
        cols_per_row = st.slider("每列顯示基本面方框數", min_value=2, max_value=4, value=3)
        render_card_grid(
            cards,
            financial_card_html,
            cols_per_row,
            button_label="詳細資訊",
            button_key_prefix="financial_detail",
            on_click=show_financial_detail,
        )
    else:
        st.info("此族群沒有可用股票代碼")

with tab4:
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.subheader("籌碼統計：三大法人最近 5 / 10 / 20 日買超")
    st.caption("買超單位為『張』；正值代表買超，負值代表賣超。資料來源使用 FinMind 三大法人買賣表。")
    g4 = st.selectbox("選擇族群（籌碼統計）", all_groups)
    subset = group_df[group_df["族群"] == g4].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    subset["名稱"] = subset["個股代碼名稱"].apply(_parse_stock_name)
    end_date = datetime.now().strftime("%Y-%m-%d")
    rows = []
    with st.spinner("讀取三大法人買賣超資料中..."):
        for _, r in subset.drop_duplicates("代碼").iterrows():
            code = r["代碼"]
            if not code:
                continue
            summary = fetch_institutional_summary(code, end_date)
            summary["個股"] = f"{r['名稱']} ({code})"
            rows.append(summary)

    if rows:
        cards = pd.DataFrame(rows).to_dict("records")
        cols_per_row = st.slider("每列顯示籌碼方框數", min_value=1, max_value=3, value=2)
        render_card_grid(cards, chip_card_html, cols_per_row)
    else:
        st.info("此族群沒有可用股票代碼")

st.markdown("---")
st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
