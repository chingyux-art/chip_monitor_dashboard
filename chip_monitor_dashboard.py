import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="台股族群監控", page_icon="📊", layout="wide")
st.title("📊 台股族群監控 App")

CSV_PATH = Path("groups.csv")


def _parse_stock_code(text: str) -> str:
    if pd.isna(text):
        return ""
    t = str(text).strip()
    m = re.search(r"\d{4,6}", t)
    return m.group(0) if m else ""


@st.cache_data(ttl=5)
def load_group_csv(path: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df


def ensure_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        lc = str(c).lower()
        if "產業" in c or "族群" in c:
            rename_map[c] = "族群"
        elif "代碼" in c and "名稱" in c:
            rename_map[c] = "個股代碼名稱"
        elif "介紹" in c or "說明" in c:
            rename_map[c] = "介紹"
    df = df.rename(columns=rename_map)
    for col in ["族群", "個股代碼名稱", "介紹"]:
        if col not in df.columns:
            df[col] = ""
    return df


@st.cache_data(ttl=900)
def fetch_ohlcv(stock_code: str, period: str = "6mo") -> pd.DataFrame:
    for suffix in [".TW", ".TWO"]:
        try:
            data = yf.download(f"{stock_code}{suffix}", period=period, interval="1d", progress=False, auto_adjust=False)
            if data is not None and len(data) > 25:
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                return data.dropna()
        except Exception:
            pass
    return pd.DataFrame()


def tech_score(stock_code: str) -> dict:
    df = fetch_ohlcv(stock_code)
    base = {"代碼": stock_code, "名稱": stock_code, "分數": 0.0, "細項": []}
    if df.empty:
        return base

    close = df["Close"].astype(float)
    vol = df["Volume"].astype(float)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    if len(macd) >= 2 and macd.iloc[-2] < 0 <= macd.iloc[-1]:
        base["分數"] += 2
        base["細項"].append("MACD 由負翻正 +2")
    if macd.iloc[-1] > signal.iloc[-1]:
        base["分數"] += 1
        base["細項"].append("MACD 快線>慢線 +1")

    short_bias = (close / close.rolling(5).mean() - 1) * 100
    long_bias = (close / close.rolling(20).mean() - 1) * 100
    if len(short_bias.dropna()) > 0 and short_bias.iloc[-1] > long_bias.iloc[-1]:
        base["分數"] += 2
        base["細項"].append("短期乖離率>長期乖離率 +2")

    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    price = close.iloc[-1]
    if price > ma5:
        base["分數"] += 1
    if price > ma10:
        base["分數"] += 1
    if price > ma20:
        base["分數"] += 1
    if ma5 > ma10 > ma20:
        base["分數"] += 1

    if len(vol) >= 4 and vol.iloc[-1] > vol.iloc[-4:-1].mean() * 2:
        base["分數"] += 1
    if vol.iloc[-1] > 500_000:
        base["分數"] += 1

    return base


@st.cache_data(ttl=3600)
def fetch_financial_score(stock_code: str) -> dict:
    out = {"代碼": stock_code, "名稱": stock_code, "分數": 0.0, "細項": []}
    t = None
    for suffix in [".TW", ".TWO"]:
        try:
            t = yf.Ticker(f"{stock_code}{suffix}")
            _ = t.info
            break
        except Exception:
            t = None
    if t is None:
        return out

    qf = t.quarterly_financials
    if isinstance(qf, pd.DataFrame) and not qf.empty:
        rev_key = next((k for k in qf.index if "Revenue" in k), None)
        gp_key = next((k for k in qf.index if "Gross Profit" in k), None)
        eps_key = next((k for k in qf.index if "Diluted EPS" in k or "Basic EPS" in k), None)

        if rev_key:
            rev = qf.loc[rev_key].dropna().astype(float).iloc[::-1]
            if len(rev) >= 3 and (rev.diff().dropna() > 0).tail(2).all():
                out["分數"] += 1
            if len(rev) >= 5:
                m = (rev.diff().dropna() > 0).tail(4).sum() * 0.2
                out["分數"] += float(m)

        if gp_key and rev_key:
            gp = qf.loc[gp_key].astype(float)
            rev = qf.loc[rev_key].astype(float)
            gm = (gp / rev.replace(0, np.nan)).dropna().iloc[::-1]
            if len(gm) >= 5:
                out["分數"] += float((gm.diff().dropna() > 0).tail(4).sum() * 0.2)

        if eps_key:
            eps = qf.loc[eps_key].dropna().astype(float).iloc[::-1]
            if len(eps) >= 2 and eps.iloc[-2] < 0 <= eps.iloc[-1]:
                out["分數"] += 1
            if len(eps) >= 3 and (eps.diff().dropna() > 0).tail(2).all():
                out["分數"] += 1

    qb = t.quarterly_balance_sheet
    if isinstance(qb, pd.DataFrame) and not qb.empty:
        k = next((x for x in qb.index if "Contract" in x and "Liab" in x), None)
        if not k:
            k = next((x for x in qb.index if "Receiv" in x), None)
        if k:
            s = qb.loc[k].dropna().astype(float).iloc[::-1]
            if len(s) >= 3 and (s.diff().dropna() > 0).tail(2).all():
                out["分數"] += 1

    return out


def latest_price_change(stock_code: str):
    df = fetch_ohlcv(stock_code, period="1mo")
    if df.empty or len(df) < 2:
        return np.nan, np.nan, np.nan
    c1, c0 = float(df["Close"].iloc[-1]), float(df["Close"].iloc[-2])
    pct = (c1 - c0) / c0 * 100
    return c1, c0, pct


if not CSV_PATH.exists():
    st.warning("找不到 groups.csv，請先放入檔案。已提供上傳功能。")
    upload = st.file_uploader("上傳族群 CSV", type=["csv"])
    if upload:
        CSV_PATH.write_bytes(upload.getvalue())
        st.success("已上傳 groups.csv，請重新整理。")
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
            "族群": g,
            "個股代碼名稱": r["個股代碼名稱"],
            "介紹": r["介紹"],
            "最新收盤": p1,
            "前一日收盤": p0,
            "漲跌幅(%)": pct,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("CSV 檔案一旦更新（mtime改變），頁面會在重新執行時自動載入最新內容。")

with tab2:
    g2 = st.selectbox("選擇族群（技術面）", all_groups)
    subset = group_df[group_df["族群"] == g2].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    results = []
    with st.spinner("計算技術面評分中..."):
        for code in subset["代碼"].dropna().unique():
            if code:
                results.append(tech_score(code))
    if results:
        out = pd.DataFrame(results).sort_values("分數", ascending=False)
        out.index = out.index + 1
        st.dataframe(out[["代碼", "分數", "細項"]], use_container_width=True)
    else:
        st.info("此族群沒有可用股票代碼")

with tab3:
    g3 = st.selectbox("選擇族群（基本面）", all_groups)
    subset = group_df[group_df["族群"] == g3].copy()
    subset["代碼"] = subset["個股代碼名稱"].apply(_parse_stock_code)
    results = []
    with st.spinner("計算基本面評分中..."):
        for code in subset["代碼"].dropna().unique():
            if code:
                results.append(fetch_financial_score(code))
    if results:
        out = pd.DataFrame(results).sort_values("分數", ascending=False)
        out.index = out.index + 1
        st.dataframe(out[["代碼", "分數", "細項"]], use_container_width=True)
    else:
        st.info("此族群沒有可用股票代碼")

st.markdown("---")
st.caption(f"更新時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
