import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf

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
st.title("📊 台股籌碼監控儀表板")
st.markdown("---")

# 側邊欄配置
st.sidebar.header("⚙️ 設定")
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
    ('2330.TW', '台積電'),
    ('2454.TW', '聯發科'),
    ('2317.TW', '鴻海'),
    ('2412.TW', '中華電'),
    ('1101.TW', '台泥'),
    ('2881.TW', '富邦金'),
    ('2882.TW', '國泰金'),
    ('2886.TW', '兆豐金'),
    ('2891.TW', '中信金'),
    ('2892.TW', '第一金'),
    ('2883.TW', '開發金'),
    ('2884.TW', '玉山金'),
    ('2885.TW', '元大金'),
    ('2880.TW', '華南金'),
    ('1303.TW', '南亞'),
    ('1301.TW', '台塑'),
    ('1326.TW', '台化'),
    ('2409.TW', '友達'),
    ('2408.TW', '奇美電'),
    ('3034.TW', '聯詠'),
]

# 生成模擬法人資料
@st.cache_data
def generate_institutional_data(start, end, stocks_list):
    """生成模擬的法人買賣超資料"""
    np.random.seed(42)
    dates = pd.date_range(start=start, end=end, freq='D')
    
    data = []
    for stock_code, stock_name in stocks_list:
        for date in dates:
            # 生成隨機的法人買賣超資料
            foreign_buy = np.random.randint(-500, 500)
            domestic_buy = np.random.randint(-300, 300)
            dealer_buy = np.random.randint(-200, 200)
            
            data.append({
                '股票代碼': stock_code.split('.')[0],
                '股票名稱': stock_name,
                '日期': date,
                '外資買超': foreign_buy,
                '內資買超': domestic_buy,
                '自營商買超': dealer_buy,
                '總買超': foreign_buy + domestic_buy + dealer_buy
            })
    
    return pd.DataFrame(data)

# 計算連買天數
def calculate_consecutive_buying_days(df):
    """計算連續買超天數"""
    consecutive_days = []
    
    for stock in df['股票代碼'].unique():
        stock_data = df[df['股票代碼'] == stock].sort_values('日期')
        stock_data['總買超'] = stock_data['外資買超'] + stock_data['內資買超'] + stock_data['自營商買超']
        
        consecutive = 0
        max_consecutive = 0
        
        for value in stock_data['總買超']:
            if value > 0:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
        
        stock_name = stock_data['股票名稱'].iloc[0]
        consecutive_days.append({
            '股票代碼': stock,
            '股票名稱': stock_name,
            '連買天數': max_consecutive
        })
    
    return pd.DataFrame(consecutive_days)

# 計算綜合評分
def calculate_score(row):
    """計算0-100的綜合評分"""
    score = 50  # 基礎分50分
    score += (row['外資買超'] / 10) * 0.3  # 外資權重30%
    score += (row['內資買超'] / 10) * 0.2  # 內資權重20%
    score += (row['自營商買超'] / 10) * 0.2  # 自營商權重20%
    score += (row['連買天數'] / 5) * 0.3  # 連買天數權重30%
    
    return max(0, min(100, score))  # 限制在0-100

# 生成K線資料
@st.cache_data
def get_kline_data(stock_code, start, end):
    """獲取K線資料"""
    try:
        df = yf.download(stock_code, start=start, end=end, progress=False)
        return df
    except:
        return None

# 加載數據
institutional_df = generate_institutional_data(start_date, end_date, STOCKS_LIST)
consecutive_buying_df = calculate_consecutive_buying_days(institutional_df)

# 匯總最新數據
latest_data = []
for stock in institutional_df['股票代碼'].unique():
    stock_data = institutional_df[institutional_df['股票代碼'] == stock].sort_values('日期')
    latest = stock_data.iloc[-1]
    stock_name = stock_data['股票名稱'].iloc[0]
    consecutive = consecutive_buying_df[consecutive_buying_df['股票代碼'] == stock]['連買天數'].values[0]
    
    total_buy = latest['外資買超'] + latest['內資買超'] + latest['自營商買超']
    
    row_data = {
        '股票代碼': stock,
        '股票名稱': stock_name,
        '外資買超': latest['外資買超'],
        '內資買超': latest['內資買超'],
        '自營商買超': latest['自營商買超'],
        '總買超': total_buy,
        '連買天數': consecutive
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
        st.metric(
            "平均外資買超",
            f"{avg_foreign:.0f}",
            f"{avg_foreign:+.0f}" if avg_foreign > 0 else f"{avg_foreign:.0f}"
        )
    
    with col2:
        avg_domestic = summary_df['內資買超'].mean()
        st.metric(
            "平均內資買超",
            f"{avg_domestic:.0f}",
            f"{avg_domestic:+.0f}" if avg_domestic > 0 else f"{avg_domestic:.0f}"
        )
    
    with col3:
        avg_dealer = summary_df['自營商買超'].mean()
        st.metric(
            "平均自營商買超",
            f"{avg_dealer:.0f}",
            f"{avg_dealer:+.0f}" if avg_dealer > 0 else f"{avg_dealer:.0f}"
        )
    
    st.markdown("---")
    
    # 法人買賣超對比
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(x=summary_df['股票名稱'], y=summary_df['外資買超'], name='外資'),
            go.Bar(x=summary_df['股票名稱'], y=summary_df['內資買超'], name='內資'),
            go.Bar(x=summary_df['股票名稱'], y=summary_df['自營商買超'], name='自營商')
        ])
        fig.update_layout(
            title="各法人買賣超對比",
            barmode='group',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 連買天數Top 10
        top_consecutive = consecutive_buying_df.nlargest(10, '連買天數')
        fig = px.bar(
            top_consecutive,
            x='股票名稱',
            y='連買天數',
            color='連買天數',
            color_continuous_scale='RdYlGn',
            title="連買天數 Top 10"
        )
        fig.update_layout(height=400, hovermode='x')
        st.plotly_chart(fig, use_container_width=True)
    
    # 最新法人籌碼表
    st.subheader("最新法人籌碼數據")
    display_df = summary_df[['股票代碼', '股票名稱', '外資買超', '內資買超', '自營商買超', '總買超']].copy()
    display_df['外資買超'] = display_df['外資買超'].apply(lambda x: f"{x:+d}")
    display_df['內資買超'] = display_df['內資買超'].apply(lambda x: f"{x:+d}")
    display_df['自營商買超'] = display_df['自營商買超'].apply(lambda x: f"{x:+d}")
    display_df['總買超'] = display_df['總買超'].apply(lambda x: f"{x:+d}")
    
    st.dataframe(display_df, use_container_width=True)

# Tab 2: 熱門股票
with tab2:
    st.header("🔥 熱門股票排行")
    
    # 按評分排序的Top 20
    top_20 = summary_df.nlargest(20, '評分').reset_index(drop=True)
    top_20['排名'] = range(1, len(top_20) + 1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 排行榜表格
        st.subheader("評分排行榜 Top 20")
        display_top20 = top_20[['排名', '股票代碼', '股票名稱', '評分', '連買天數', '總買超']].copy()
        display_top20['評分'] = display_top20['評分'].apply(lambda x: f"{x:.1f}")
        display_top20['總買超'] = display_top20['總買超'].apply(lambda x: f"{x:+d}")
        
        st.dataframe(display_top20, use_container_width=True, hide_index=True)
    
    with col2:
        # 評分分布
        fig = px.histogram(
            top_20,
            x='評分',
            nbins=20,
            color='評分',
            color_continuous_scale='Viridis',
            title="評分分布"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 評分vs連買天數散點圖
    fig = px.scatter(
        top_20,
        x='連買天數',
        y='評分',
        size='總買超',
        hover_name='股票名稱',
        color='評分',
        color_continuous_scale='RdYlGn',
        title="評分 vs 連買天數",
        labels={'連買天數': '連買天數 (天)', '評分': '綜合評分 (0-100)'}
    )
    fig.update_layout(height=500, hovermode='closest')
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: K線分析
with tab3:
    st.header("📈 K線分析")
    
    # 選擇股票
    selected_stocks = st.multiselect(
        "選擇股票",
        options=[f"{row['股票代碼']} {row['股票名稱']}" for _, row in summary_df.iterrows()],
        default=[f"{summary_df.iloc[0]['股票代碼']} {summary_df.iloc[0]['股票名稱']}"],
        max_selections=3
    )
    
    for selected in selected_stocks:
        stock_code_display = selected.split()[0]
        stock_name = selected.split()[1]
        stock_code = f"{stock_code_display}.TW"
        
        st.subheader(f"{stock_name} ({stock_code_display})")
        
        kline_data = get_kline_data(stock_code, start_date, end_date)
        
        if kline_data is not None and len(kline_data) > 0:
            # 創建K線圖
            fig = go.Figure(data=[go.Candlestick(
                x=kline_data.index,
                open=kline_data['Open'],
                high=kline_data['High'],
                low=kline_data['Low'],
                close=kline_data['Close'],
                name=stock_name
            )])
            
            fig.update_layout(
                title=f"{stock_name} K線圖",
                yaxis_title="股價 (TWD)",
                xaxis_title="日期",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 顯示基本統計
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("最高價", f"${kline_data['High'].max():.2f}")
            with col2:
                st.metric("最低價", f"${kline_data['Low'].min():.2f}")
            with col3:
                st.metric("收盤價", f"${kline_data['Close'].iloc[-1]:.2f}")
            with col4:
                change = ((kline_data['Close'].iloc[-1] - kline_data['Open'].iloc[0]) / kline_data['Open'].iloc[0]) * 100
                st.metric("漲跌幅", f"{change:.2f}%")
        else:
            st.warning(f"無法取得 {stock_name} 的K線資料")
        
        st.markdown("---")

# Tab 4: 詳細資料
with tab4:
    st.header("📋 詳細資料")
    
    # 完整表格
    st.subheader("完整數據表")
    full_display_df = summary_df[['股票代碼', '股票名稱', '外資買超', '內資買超', '自營商買超', '總買超', '連買天數', '評分']].copy()
    full_display_df = full_display_df.sort_values('評分', ascending=False).reset_index(drop=True)
    full_display_df.index = full_display_df.index + 1
    
    st.dataframe(full_display_df, use_container_width=True)
    
    # 下載CSV
    csv = full_display_df.to_csv(index=False)
    st.download_button(
        label="📥 下載數據 (CSV)",
        data=csv,
        file_name=f"chip_monitor_data_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )
    
    # 統計摘要
    st.subheader("統計摘要")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("監控股票數", len(summary_df))
    with col2:
        st.metric("買超股票數", len(summary_df[summary_df['總買超'] > 0]))
    with col3:
        st.metric("賣超股票數", len(summary_df[summary_df['總買超'] < 0]))
    with col4:
        st.metric("平均評分", f"{summary_df['評分'].mean():.1f}")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>📊 台股籌碼監控儀表板 | 最後更新: {}</p>
    <p style='font-size: 12px; color: #888;'>本平台數據為模擬數據，僅供參考使用</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)
