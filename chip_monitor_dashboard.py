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

# 設置頁面配置
st.set_page_config(
    page_title="台股籌碼監控儀表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加CSS樣式
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .positive {
        color: #d32f2f;
    }
    .negative {
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

# 標題
st.title("📊 台股籌碼監控儀表板 v2.0")
st.markdown("---")

# 側邊欄配置
st.sidebar.header("⚙️ 設定")
data_source = st.sidebar.radio(
    "選擇數據源",
    options=["真實數據 (TWSE)", "模擬數據 (測試用)"],
    help="真實數據來自台灣證券交易所"
)

start_date = st.sidebar.date_input(
    "開始日期",
    value=datetime.now() - timedelta(days=30),
    max_value=datetime.now()
)
end_date = st.sidebar.date_input(
    "結束日期",
    value=datetime.now(),
    max_value=datetime.now()
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

# 創建會話以支持重試
def create_session_with_retries():
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5, status_forcelist=(500, 502, 504))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# 從TWSE抓取籌碼數據
@st.cache_data(ttl=3600)
def fetch_real_institutional_data(stock_code, start, end):
    """從TWSE API抓取真實法人買賣超數據"""
    try:
        session = create_session_with_retries()
        data_list = []
        
        current_date = start
        while current_date <= end:
            date_str = current_date.strftime('%Y%m%d')
            url = f"https://www.twse.com.tw/rpt/t05020.php?date={date_str}&response=json&selectType=ALLBUT0999"
            
            try:
                response = session.get(url, timeout=10)
                response.encoding = 'utf-8'
                
                if response.status_code == 200:
                    json_data = response.json()
                    if 'data' in json_data:
                        for row in json_data['data']:
                            if row[0].strip() == stock_code:
                                try:
                                    data_list.append({
                                        '日期': current_date,
                                        '股票代碼': stock_code,
                                        '外資買超': int(row[4].replace(',', '')),
                                        '內資買超': int(row[5].replace(',', '')),
                                        '自營商買超': int(row[6].replace(',', ''))
                                    })
                                except (ValueError, IndexError):
                                    pass
            except Exception:
                pass
            
            current_date += timedelta(days=1)
            time.sleep(0.1)
        
        if data_list:
            return pd.DataFrame(data_list)
        else:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# 生成模擬法人資料
@st.cache_data(ttl=3600)
def generate_institutional_data(start, end, stocks_list):
    """生成模擬的法人買賣超資料"""
    np.random.seed(42)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    data = []
    for stock_code, stock_name in stocks_list:
        for date in dates:
            foreign_buy = np.random.randint(-500, 500)
            domestic_buy = np.random.randint(-300, 300)
            dealer_buy = np.random.randint(-200, 200)
            
            data.append({
                '股票代碼': stock_code,
                '股票名稱': stock_name,
                '日期': date,
                '外資買超': foreign_buy,
                '內資買超': domestic_buy,
                '自營商買超': dealer_buy,
            })
    
    return pd.DataFrame(data)

# 計算連買天數
def calculate_buy_streaks(df):
    out = []
    for code in df['股票代碼'].unique():
        s = df[df['股票代碼'] == code].sort_values('日期').copy()
        s['總買超'] = s['外資買超'] + s['內資買超'] + s['自營商買超']

        # max streak
        consec = 0
        max_streak = 0
        for v in s['總買超']:
            if v > 0:
                consec += 1
                max_streak = max(max_streak, consec)
            else:
                consec = 0

        # current streak (from latest backwards)
        current_streak = 0
        for v in reversed(s['總買超'].tolist()):
            if v > 0:
                current_streak += 1
            else:
                break

        name = s['股票名稱'].iloc[0] if '股票名稱' in s.columns else code
        out.append({'股票代碼': code, '股票名稱': name,
                    '最大連買天數': max_streak, '最新連買天數': current_streak})
    return pd.DataFrame(out)

# 計算綜合評分
def calculate_score(row):
    """計算0-100的綜合評分"""
    score = 50
    score += (row['外資買超'] / 10) * 0.3
    score += (row['內資買超'] / 10) * 0.2
    score += (row['自營商買超'] / 10) * 0.2
    score += (row['連買天數'] / 5) * 0.3
    
    return max(0, min(100, score))

# 生成K線資料
@st.cache_data(ttl=3600)
def get_kline_data(stock_code, start, end):
    """獲取K線資料"""
    try:
        ticker = f"{stock_code}.TW"
        df = yf.download(ticker, start=start, end=end, progress=False)
        return df if len(df) > 0 else None
    except Exception:
        return None

# 加載數據
st.info(f"📊 數據源: {data_source} | 日期範圍: {start_date} 至 {end_date}")

if data_source == "真實數據 (TWSE)":
    st.warning("⚠️ 正在從TWSE獲取真實數據，這可能需要1-2分鐘...")
    with st.spinner("正在加載數據..."):
        all_data = []
        for stock_code, stock_name in STOCKS_LIST:
            real_data = fetch_real_institutional_data(stock_code, start_date, end_date)
            if not real_data.empty:
                real_data['股票名稱'] = stock_name
                all_data.append(real_data)
        
        if all_data:
            institutional_df = pd.concat(all_data, ignore_index=True)
        else:
            st.warning("⚠️ 無法獲取真實數據，自動切換到模擬數據")
            institutional_df = generate_institutional_data(start_date, end_date, STOCKS_LIST)
else:
    institutional_df = generate_institutional_data(start_date, end_date, STOCKS_LIST)

# 添加總買超列
institutional_df['總買超'] = (
    institutional_df['外資買超'] + 
    institutional_df['內資買超'] + 
    institutional_df['自營商買超']
)

consecutive_buying_df = calculate_consecutive_buying_days(institutional_df)

# 匯總最新數據
latest_data = []
for stock_code, stock_name in STOCKS_LIST:
    stock_data = institutional_df[institutional_df['股票代碼'] == stock_code].sort_values('日期')
    
    if len(stock_data) > 0:
        latest = stock_data.iloc[-1]
        consecutive = consecutive_buying_df[consecutive_buying_df['股票代碼'] == stock_code]['連買天數'].values
        
        row_data = {
            '股票代碼': stock_code,
            '股票名稱': stock_name,
            '外資買超': int(latest['外資買超']),
            '內資買超': int(latest['內資買超']),
            '自營商買超': int(latest['自營商買超']),
            '總買超': int(latest['總買超']),
            '連買天數': int(consecutive[0]) if len(consecutive) > 0 else 0
        }
        row_data['評分'] = calculate_score(row_data)
        latest_data.append(row_data)

summary_df = pd.DataFrame(latest_data)

# 創建Tab
tab1, tab2, tab3, tab4 = st.tabs(["法人籌碼", "熱門股票", "K線分析", "詳細資料"])

# Tab 1: 法人籌碼
with tab1:
    st.header("📈 法人籌碼分析")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_foreign = summary_df['外資買超'].mean()
        st.metric("平均外資買超", f"{avg_foreign:.0f}", f"{avg_foreign:+.0f}" if avg_foreign > 0 else f"{avg_foreign:.0f}")
    
    with col2:
        avg_domestic = summary_df['內資買超'].mean()
        st.metric("平均內資買超", f"{avg_domestic:.0f}", f"{avg_domestic:+.0f}" if avg_domestic > 0 else f"{avg_domestic:.0f}")
    
    with col3:
        avg_dealer = summary_df['自營商買超'].mean()
        st.metric("平均自營商買超", f"{avg_dealer:.0f}", f"{avg_dealer:+.0f}" if avg_dealer > 0 else f"{avg_dealer:.0f}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(x=summary_df['股票名稱'], y=summary_df['外資買超'], name='外資'),
            go.Bar(x=summary_df['股票名稱'], y=summary_df['內資買超'], name='內資'),
            go.Bar(x=summary_df['股票名稱'], y=summary_df['自營商買超'], name='自營商')
        ])
        fig.update_layout(title="各法人買賣超對比", barmode='group', height=400, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if len(consecutive_buying_df) > 0:
            top_consecutive = consecutive_buying_df.nlargest(10, '連買天數')
            fig = px.bar(top_consecutive, x='股票名稱', y='連買天數', color='連買天數', title="連買天數 Top 10", color_continuous_scale='Viridis')
            fig.update_layout(height=400, hovermode='x')
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("最新法人籌碼數據")
    display_df = summary_df[['股票代碼', '股票名稱', '外資買超', '內資買超', '自營商買超', '總買超']].copy()
    display_df['外資買超'] = display_df['外資買超'].apply(lambda x: f"{x:+d}")
    display_df['內資買超'] = display_df['內資買超'].apply(lambda x: f"{x:+d}")
    display_df['自營商買超'] = display_df['自營商買超'].apply(lambda x: f"{x:+d}")
    display_df['總買超'] = display_df['總買超'].apply(lambda x: f"{x:+d}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# Tab 2: 熱門股票
with tab2:
    st.header("🔥 熱門股票排行")
    
    top_20 = summary_df.nlargest(20, '評分').reset_index(drop=True)
    top_20['排名'] = range(1, len(top_20) + 1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("評分排行榜 Top 20")
        display_top20 = top_20[['排名', '股票代碼', '股票名稱', '評分', '連買天數', '總買超']].copy()
        display_top20['評分'] = display_top20['評分'].apply(lambda x: f"{x:.1f}")
        display_top20['總買超'] = display_top20['總買超'].apply(lambda x: f"{x:+d}")
        st.dataframe(display_top20, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("評分分布")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=top_20['評分'], nbinsx=15, name='評分', marker_color='rgb(55, 83, 109)', opacity=0.7))
        fig.update_layout(title="評分分布", xaxis_title="評分 (0-100)", yaxis_title="股票數量", height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("評分 vs 連買天數")
    top_20_plot = top_20.copy()
    top_20_plot['BubbleSize'] = top_20_plot['總買超'].abs()
    fig = px.scatter(
        top_20_plot,
        x='連買天數',
        y='評分',
        size='BubbleSize',
        hover_name='股票名稱',
        color='評分',
        color_continuous_scale='RdYlGn',
        title="評分 vs 連買天數分析",
        size_max=40
    )
    fig.update_layout(height=500, hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: K線分析
with tab3:
    st.header("📈 K線分析")
    
    selected_stocks = st.multiselect(
        "選擇股票",
        options=[f"{row['股票代碼']} {row['股票名稱']}" for _, row in summary_df.iterrows()],
        default=[f"{summary_df.iloc[0]['股票代碼']} {summary_df.iloc[0]['股票名稱']}"],
        max_selections=3
    )
    
    if selected_stocks:
        for selected in selected_stocks:
            stock_code_display = selected.split()[0]
            stock_name = selected.split()[1]
            
            st.subheader(f"{stock_name} ({stock_code_display})")
            
            kline_data = get_kline_data(stock_code_display, start_date, end_date)
            
            if kline_data is not None and len(kline_data) > 0:
                fig = go.Figure(data=[go.Candlestick(
                    x=kline_data.index, open=kline_data['Open'], high=kline_data['High'],
                    low=kline_data['Low'], close=kline_data['Close'], name=stock_name
                )])
                fig.update_layout(title=f"{stock_name} K線圖", yaxis_title="股價 (TWD)", xaxis_title="日期", template="plotly_white", height=500, hovermode='x unified')
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
                        change = ((kline_data['Close'].iloc[-1] - kline_data['Open'].iloc[0]) / kline_data['Open'].iloc[0]) * 100
                        st.metric("漲跌幅", f"{change:.2f}%")
                    except:
                        st.metric("漲跌幅", "N/A")
            else:
                st.warning(f"⚠️ 無法取得 {stock_name} 的K線資料")
            
            st.markdown("---")
    else:
        st.info("👈 請從左側選擇要分析的股票")

# Tab 4: 詳細資料
with tab4:
    st.header("📋 詳細資料")
    
    st.subheader("完整數據表")
    full_display_df = summary_df[['股票代碼', '股票名稱', '外資買超', '內資買超', '自營商買超', '總買超', '連買天數', '評分']].copy()
    full_display_df = full_display_df.sort_values('評分', ascending=False).reset_index(drop=True)
    full_display_df.index = full_display_df.index + 1
    
    display_full = full_display_df.copy()
    display_full['外資買超'] = display_full['外資買超'].apply(lambda x: f"{x:+d}")
    display_full['內資買超'] = display_full['內資買超'].apply(lambda x: f"{x:+d}")
    display_full['自營商買超'] = display_full['自營商買超'].apply(lambda x: f"{x:+d}")
    display_full['總買超'] = display_full['總買超'].apply(lambda x: f"{x:+d}")
    display_full['評分'] = display_full['評分'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(display_full, use_container_width=True)
    
    csv = full_display_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(label="📥 下載數據 (CSV)", data=csv, file_name=f"chip_monitor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    
    st.subheader("📊 統計摘要")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("監控股票數", len(summary_df))
    with col2:
        st.metric("買超股票數", len(summary_df[summary_df['總買超'] > 0]))
    with col3:
        st.metric("賣超股票數", len(summary_df[summary_df['總買超'] < 0]))
    with col4:
        st.metric("平均評分", f"{summary_df['評分'].mean():.1f}")
    with col5:
        st.metric("最高評分", f"{summary_df['評分'].max():.1f}")
    
    st.subheader("📈 評分統計分析")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=summary_df['評分'], nbinsx=15, marker_color='rgb(158, 202, 225)', name='股票數量'))
        fig.update_layout(title="全部股票評分分布", xaxis_title="評分", yaxis_title="股票數量", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        buy_count = len(summary_df[summary_df['總買超'] > 0])
        sell_count = len(summary_df[summary_df['總買超'] < 0])
        neutral_count = len(summary_df[summary_df['總買超'] == 0])
        
        fig = go.Figure(data=[go.Pie(labels=['買超', '賣超', '平手'], values=[buy_count, sell_count, neutral_count], marker=dict(colors=['#d32f2f', '#1976d2', '#90a4ae']))])
        fig.update_layout(title="法人買賣超分布", height=400)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 關於此應用\n- **數據源**: TWSE\n- **K線數據**: Yahoo Finance\n- **更新頻率**: 每日營業時間")

with col2:
    st.markdown("### 🎯 評分說明\n- **外資**: 30%\n- **內資**: 20%\n- **自營商**: 20%\n- **連買天數**: 30%")

with col3:
    st.markdown("### ⚠️ 免責聲明\n本平台僅供參考使用，不構成投資建議。股票投資存在風險，請理性投資。")

st.markdown(f"""<div style='text-align: center; color: #888; font-size: 12px;'>
    最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | v2.0 | Real API Integration
</div>""", unsafe_allow_html=True)
