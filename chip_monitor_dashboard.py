import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="台股族群監控", page_icon="📊", layout="wide")
st.title("📊 台股族群監控 App")

CSV_PATH = Path("Group.csv")


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


if not CSV_PATH.exists():
    st.error("找不到 Group.csv，請確認檔案位於專案根目錄 chingyux-art/chip_monitor_dashboard/Group.csv。")
    st.stop()

raw_df = load_group_csv(str(CSV_PATH), CSV_PATH.stat().st_mtime)
group_df = ensure_group_columns(raw_df)
all_groups = sorted([g for g in group_df["族群"].dropna().unique() if str(g).strip()])

if not all_groups:
    st.error("CSV 缺少有效的『族群』資料")
    st.stop()

tab1, tab2, tab3 = st.tabs(["分頁一：族群與股價", "分頁二：技術面評分", "分頁三：基本面評分"])

with tab1:
    g = st.selectbox("選擇族群", all_groups)
    subset = group_df[group_df["族群"] == g].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    rows = []
    for _, r in subset.iterrows():
        code = r["代碼"]
        p1, p0, pct = latest_price_change(code) if code else (np.nan, np.nan, np.nan)
        rows.append({
            "產業分類": g,
            "個股代碼名稱": r["個股代碼名稱"],
            "產業定位": r["產業定位"],
            "公司業務說明": r["公司業務說明"],
            "在該產業的地位與優勢": r["在該產業的地位與優勢"],
            "最新收盤": p1,
            "前一日收盤": p0,
            "漲跌幅(%)": pct,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab2:
    g2 = st.selectbox("選擇族群（技術面）", all_groups)
    subset = group_df[group_df["族群"] == g2].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    subset["名稱"] = subset["個股代碼名稱"].apply(_parse_stock_name)
    name_map = dict(zip(subset["代碼"], subset["名稱"]))
    results = []
    with st.spinner("計算技術面評分中..."):
        for code in subset["代碼"].dropna().unique():
            if code:
                s = tech_score(code)
                s["個股名稱"] = name_map.get(code, code)
                results.append(s)
    if results:
        out = pd.DataFrame(results).sort_values("分數", ascending=False)
        out["個股"] = out["個股名稱"] + " (" + out["代碼"] + ")"
        st.dataframe(out[["個股", "分數", "細項"]], use_container_width=True, hide_index=True)

        chosen = st.selectbox("選擇個股查看近一年K線", out["個股"].tolist())
        selected_code = re.search(r"\((\d{4,6})\)", chosen).group(1)
        kdf = fetch_ohlcv(selected_code, period="1y")
        if not kdf.empty:
            fig = go.Figure(data=[go.Candlestick(x=kdf.index, open=kdf["Open"], high=kdf["High"], low=kdf["Low"], close=kdf["Close"])])
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, title=f"{chosen} 近一年K線")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("此族群沒有可用股票代碼")

with tab3:
    g3 = st.selectbox("選擇族群（基本面）", all_groups)
    subset = group_df[group_df["族群"] == g3].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    subset["名稱"] = subset["個股代碼名稱"].apply(_parse_stock_name)
    name_map = dict(zip(subset["代碼"], subset["名稱"]))
    results = []
    with st.spinner("計算基本面評分中..."):
        for code in subset["代碼"].dropna().unique():
            if code:
                s = fetch_financial_score(code)
                s["個股名稱"] = name_map.get(code, code)
                results.append(s)
    if results:
        out = pd.DataFrame(results).sort_values("分數", ascending=False)
        out["個股"] = out["個股名稱"] + " (" + out["代碼"] + ")"
        st.dataframe(out[["個股", "分數", "細項"]], use_container_width=True, hide_index=True)
    else:
        st.info("此族群沒有可用股票代碼")

st.markdown("---")
st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
