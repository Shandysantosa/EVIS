import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout, Input
from tensorflow.keras.layers import Conv1D, LSTM, Flatten
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
import psutil
from tensorflow.keras.callbacks import EarlyStopping

# Set page layout to centered
st.set_page_config(page_title="CO2 Emissions Analytiucs App", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for background image with gradient overlay
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)), url("https://images.unsplash.com/photo-1578991132108-16c5296b63dc?w=900&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YXV0b21vdGl2ZXxlbnwwfHwwfHx8MA%3D%3D");
    background-size: cover; /* Cover the entire container */
    background-position: center; /* Center the image */
    background-repeat: no-repeat; /* Do not repeat the image */
    background-attachment: scroll; /* Allow scrolling */
}

[data-testid="stSidebar"] {
    background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url("https://images.unsplash.com/photo-1557683316-973673baf926");
    background-size: cover; /* Cover the entire container */
    background-position: center; /* Center the image */
    background-repeat: no-repeat; /* Do not repeat the image */
    background-attachment: scroll; /* Allow scrolling */
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Add custom CSS
st.markdown("""
<style>
    /* 主题颜色 */
    :root {
        --primary-bg: #0D1117;
        --secondary-bg: #161B22;
        --accent-color: #FFFFFF;
        --text-color: #E6EDF3;
        --card-bg: rgba(22, 27, 34, 0.95);
        --success-color: #3FB950;
        --warning-color: #F85149;
        --chart-blue: #4589FF;
        --chart-grid: rgba(69, 137, 255, 0.1);
        --border-color: rgba(69, 137, 255, 0.15);
        --card-padding: 1.5rem;
        --border-radius: 4px;
    }

    /* 全局样式 */
    .stApp {
        background: var(--primary-bg);
        color: var(--text-color);
        font-family: 'Roboto Mono', monospace;
    }

    /* 基础卡片样式 */
    .base-card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: var(--card-padding);
        margin: 1rem 0;
        position: relative;
    }

    /* 装饰角样式 */
    .corner-decoration::before, .corner-decoration::after {
        content: '';
        position: absolute;
        width: 8px;
        height: 8px;
    }

    .corner-decoration::before {
        top: 0;
        right: 0;
        border-top: 2px solid var(--accent-color);
        border-right: 2px solid var(--accent-color);
    }

    .corner-decoration::after {
        bottom: 0;
        left: 0;
        border-bottom: 2px solid var(--accent-color);
        border-left: 2px solid var(--accent-color);
    }

    /* 卡片样式继承 */
    .css-card, .metric-card, .plot-container {
        composes: base-card corner-decoration;
    }

    /* 标题样式 */
    h1, h2, h3 {
        color: var(--text-color);
        font-weight: 500;
        margin-bottom: 1rem;
        font-family: 'Roboto Mono', monospace;
        letter-spacing: 0.5px;
    }

    /* 描述文本样式 */
    .description-text {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 0.9rem;
        margin-bottom: 1rem;
        letter-spacing: 0.3px;
    }

    /* 指标卡片样式 */
    .metric-card {
        padding: var(--card-padding);
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--accent-color);
        opacity: 0.5;
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--accent-color);
        margin: 0.5rem 0;
        font-family: 'Roboto Mono', monospace;
    }

    .metric-label {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: center;
    }

    .metric-unit {
        font-size: 0.8rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }

    /* 状态指示器样式 */
    .status-indicator {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        padding: 0.5rem;
        background: var(--secondary-bg);
        border-radius: var(--border-radius);
    }

    .status-text {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-value {
        font-weight: 500;
        padding: 0.25rem 0.75rem;
        border-radius: var(--border-radius);
        font-size: 0.9rem;
    }

    .status-value.success {
        color: var(--success-color);
        background: rgba(63, 185, 80, 0.1);
    }

    .status-value.warning {
        color: var(--warning-color);
        background: rgba(248, 81, 73, 0.1);
    }

    /* 图表容器样式 */
    .plot-container {
        margin: 1.5rem 0;
    }

    .chart-description {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1rem 0;
        padding: 0.5rem;
        background: var(--secondary-bg);
        border-radius: var(--border-radius);
    }

    .chart-icon {
        font-size: 1.2rem;
    }

    .chart-text {
        font-size: 0.9rem;
        opacity: 0.8;
        letter-spacing: 0.3px;
    }

    /* 数据表格样式 */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
    }

    .dataframe th, .dataframe td {
        padding: 0.75rem;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }

    .dataframe th {
        background: rgba(69, 137, 255, 0.1);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* 控件样式统一 */
    .stSelectbox > div, .stSlider > div {
        background: var(--secondary-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--border-radius);
        padding: 0.5rem;
    }

    .stSelectbox > div:hover, .stSlider > div:hover {
        border-color: var(--accent-color);
    }
</style>
""", unsafe_allow_html=True)

if "transformer_model" not in st.session_state:
    st.session_state.transformer_model = None
if "cnn_lstm_model" not in st.session_state:
    st.session_state.cnn_lstm_model = None

# scalers
if "scaler_X_transformer" not in st.session_state:
    st.session_state.scaler_X_transformer = None
if "scaler_y_transformer" not in st.session_state:
    st.session_state.scaler_y_transformer = None
if "scaler_X_cnn" not in st.session_state:
    st.session_state.scaler_X_cnn = None
if "scaler_y_cnn" not in st.session_state:
    st.session_state.scaler_y_cnn = None




# Helper function: Is it winter in Taiwan?
def is_winter_in_taiwan(modified_date):
    return modified_date.month in [12, 1, 2]

# Add this function to calculate carbon tax
def calculate_carbon_tax(co2_emissions, tax_rate_per_ton):
    return co2_emissions * tax_rate_per_ton / 1000000  # Convert emissions from g/km to tons/km

# Load data
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)
    df['modifiedOn'] = pd.to_datetime(
        df['modifiedOn'].str[:-4], errors='coerce', format="%a %b %d %H:%M:%S %Z %Y"
    ).dt.tz_localize('America/New_York')

    df['adjusted_co2TailpipeGkm_converted'] = df.apply(
        lambda row: row['co2TailpipeGkm_converted'] * 1.1
        if is_winter_in_taiwan(row['modifiedOn'])
        else row['co2TailpipeGkm_converted'],
        axis=1
    )
    return df

# Global Dataset Path
FILE_PATH = './Vehicles_10-24.csv'

# Load data
data = load_data(FILE_PATH)

def main():
    # Page title and description
    st.markdown("""
        <div style='text-align: center;'>
            <h1>CO₂ Emissions Analytics Dashboard</h1>
            <p style='font-size: 1.2rem; color: var(--text-color); opacity: 0.8; margin-bottom: 2rem;'>
                Advanced real-time analytics and AI-powered predictions for vehicle emissions
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.title("Enter Emission Data")
    page = st.sidebar.selectbox("Choose a page", ["Data Visualization & Forecast", "Personal CO2 Calculator"])
    
    if page == "Data Visualization & Forecast":
        visualization_page(data)
    elif page == "Personal CO2 Calculator":
        upload_predict_page()

def visualization_page(df):
    
    st.sidebar.header("Filter 🛠️")
    
    # Filter Options
    years = df['year'].dropna().unique()
    makes = df['make'].dropna().unique()
    vehicle_classes = df['VClass'].dropna().unique()
    
    selected_years = st.sidebar.multiselect("Select Year(s)", options=["All"] + sorted(years), default=["All"])
    selected_makes = st.sidebar.multiselect("Select Make(s)", options=["All"] + sorted(makes), default=["All"])
    selected_classes = st.sidebar.multiselect("Select Vehicle Class(es)", options=["All"] + sorted(vehicle_classes), default=["All"])
    
    if "All" in selected_years:
        selected_years = years
    if "All" in selected_makes:
        selected_makes = makes
    if "All" in selected_classes:
        selected_classes = vehicle_classes

    # Engine Size Filter
    min_displ, max_displ = st.sidebar.slider(
        "Select Engine Size Range (L)",
        float(df['displ'].min()), float(df['displ'].max()),
        (float(df['displ'].min()), float(df['displ'].max()))
    )

    # Apply Filters
    filtered_df = df[
        (df['year'].isin(selected_years)) &
        (df['make'].isin(selected_makes)) &
        (df['VClass'].isin(selected_classes)) &
        (df['displ'].between(min_displ, max_displ))
    ]

    # Display key metrics
    st.markdown("""
        <div style='text-align: center;'>
            <h2>Real-time Metrics</h2>
            <p style='color: var(--text-color); opacity: 0.8; margin-bottom: 1rem;'>
                Key indicators based on current filters
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Get system metrics
    system_load = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    api_status = "Online"  # Placeholder for actual API check

    # Determine colors
    system_load_color = "var(--success-color)" if system_load < 50 else "var(--warning-color)" if system_load < 80 else "var(--error-color)"
    memory_usage_color = "var(--success-color)" if memory_usage < 50 else "var(--warning-color)" if memory_usage < 80 else "var(--error-color)"

    col1, col2, col3, col4, col5 = st.columns(5)

    # Card style
    card_style = """
        background-color: rgba(19, 33, 66, 0.9);
        border: 1px solid #4fb2f2;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
        color: white;
        text-align: center;
    """
    
    with col1:
        avg_co2 = filtered_df['co2TailpipeGkm_converted'].mean() if 'co2TailpipeGkm_converted' in filtered_df.columns else 0
        st.markdown(f"""
            <div style="{card_style}">
                <div class='metric-value'>{avg_co2:.1f}</div>
                <div class='metric-label'>Average CO₂ Emissions</div>
                <div class='metric-unit'>g/km</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        avg_displ = filtered_df['displ'].mean() if 'displ' in filtered_df.columns else 0
        st.markdown(f"""
            <div style="{card_style}">
                <div class='metric-value'>{avg_displ:.1f}</div>
                <div class='metric-label'>Average Displacement</div>
                <div class='metric-unit'>L</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        model_count = len(filtered_df)
        st.markdown(f"""
            <div style="{card_style}">
                <div class='metric-value'>{model_count:,}</div>
                <div class='metric-label'>Total Models</div>
                <div class='metric-unit'>units</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        manufacturer_count = len(filtered_df['make'].unique())
        st.markdown(f"""
            <div style="{card_style}">
                <div class='metric-value'>{manufacturer_count:,}</div>
                <div class='metric-label'>Manufacturers</div>
                <div class='metric-unit'>companies</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div style="{card_style}">
                <div style='margin: 1rem 0; text-align: center;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span>System Load</span>
                        <span style='color: {system_load_color};'>{system_load}%</span>
                    </div>
                    <div style='height: 4px; background: rgba(255,255,255,0.2); border-radius: 2px;'>
                        <div style='width: {system_load}%; height: 100%; background: {system_load_color}; border-radius: 2px;'></div>
                    </div>
                </div>
                <div style='margin: 1rem 0; text-align: center;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span>Memory Usage</span>
                        <span style='color: {memory_usage_color};'>{memory_usage}%</span>
                    </div>
                    <div style='height: 4px; background: rgba(255,255,255,0.2); border-radius: 2px;'>
                        <div style='width: {memory_usage}%; height: 100%; background: {memory_usage_color}; border-radius: 2px;'></div>
                    </div>
                </div>
                <div style='margin: 1rem 0; text-align: center;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                        <span>API Status</span>
                        <span style='color: var(--success-color);'>{api_status}</span>
                    </div>
                    <div style='height: 4px; background: rgba(255,255,255,0.2); border-radius: 2px;'>
                        <div style='width: 100%; height: 100%; background: var(--success-color); border-radius: 2px;'></div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)


    col1, col2 = st.columns(2)
    with col1:
        # Top 10 Vehicles with Highest CO2 Emissions
        st.header("Top 10 Vehicles with Highest CO2 Emissions 🚗💨")
        top10 = filtered_df.sort_values(by='co2TailpipeGkm_converted', ascending=False).head(10)
        st.dataframe(
            top10[['year', 'make', 'model', 'VClass', 'displ', 'co2TailpipeGkm_converted']],
            column_config={
                'year': st.column_config.NumberColumn(format="%d")
            }
        )

    with col2:
        st.markdown("""
            <div class='plot-container animated'>
                <h3>Vehicle Type Emissions</h3>
                <div class='chart-description'>
                    <span class='chart-icon'>🔄</span>
                    <span class='chart-text'>Emissions distribution by vehicle type</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if 'VClass' in filtered_df.columns and 'co2TailpipeGkm_converted' in filtered_df.columns:
            class_emissions = filtered_df.groupby('VClass')['co2TailpipeGkm_converted'].mean().reset_index()
            fig_pie = go.Figure(data=[go.Pie(
                labels=class_emissions['VClass'],
                values=class_emissions['co2TailpipeGkm_converted'],
                hole=0.6
            )])
            fig_pie.update_layout(
                plot_bgcolor='rgba(11, 20, 55, 0.3)',
                paper_bgcolor='rgba(11, 20, 55, 0.3)',
                font_color='#E6EDF3',
                font_family='Arial',
                height=400  # Set chart height
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("Vehicle class or CO₂ emissions data not available")


    # CO₂ Emissions Histogram
    st.header("Histogram of CO₂ Emissions 📊")
    st.plotly_chart(px.histogram(filtered_df, x='co2TailpipeGkm_converted', nbins=30,
                                  title="CO₂ Emissions Distribution",
                                  labels={'co2TailpipeGkm_converted': 'CO₂ Emissions (g/km)'}))

    
    # Average CO2 Emissions Over Years
    st.header("Average CO₂ Emissions Over Years 📈")
    avg_co2_year = filtered_df.groupby('year')['co2TailpipeGkm_converted'].mean().reset_index()
    line_chart = px.line(avg_co2_year, x='year', y='co2TailpipeGkm_converted',
                         title='Average CO₂ Emissions Over Years',
                         labels={'year': 'Year', 'co2TailpipeGkm_converted': 'CO₂ Emissions (g/km)'})
    st.plotly_chart(line_chart)

    col1, col2 = st.columns(2)
    with col1:
        # 3D Scatter Plot: Engine Size, Fuel Economy, CO₂ Emissions
        st.header("3D Analysis of CO₂ Emissions 🌐")
        st.plotly_chart(px.scatter_3d(filtered_df, x='displ', y='comb08', z='co2TailpipeGkm_converted',
                                    color='make', title="3D Scatter: Engine Size, Fuel Economy, CO₂ Emissions",
                                    labels={'displ': 'Engine Size (L)', 'comb08': 'Combined Fuel Economy (mpg)', 'co2TailpipeGkm_converted': 'CO₂ Emissions (g/km)'}))
    with col2:
        # Engine Size vs CO2 Emissions
        st.header("Engine Size vs CO₂ Emissions")
        scatter_chart = px.scatter(filtered_df, x='displ', y='co2TailpipeGkm_converted',
                                title='Engine Size vs CO₂ Emissions',
                                labels={'displ': 'Engine Displacement (L)', 'co2TailpipeGkm_converted': 'CO₂ Emissions (g/km)'},
                                trendline='ols')
        st.plotly_chart(scatter_chart)

    # Forecasting
    forecast_co2_emissions(data)

    







# Sample DataFrame structure for testing
def prepare_sequence_data(df, seq_len=5, target='co2TailpipeGkm_converted'):
    df = df.sort_values('year')
    features = ['displ', 'city08', 'highway08', 'comb08']
    X_seq, y_seq, years = [], [], []

    grouped = df.groupby('year')
    years_available = sorted(grouped.groups.keys())

    for i in range(len(years_available) - seq_len):
        window_years = years_available[i:i+seq_len+1]
        window_df = df[df['year'].isin(window_years)]

        if len(window_df['year'].unique()) < seq_len + 1:
            continue

        feature_seq = []
        for yr in window_years[:-1]:
            year_data = window_df[window_df['year'] == yr][features].mean().values
            feature_seq.append(year_data)

        target_year = window_years[-1]
        target_data = window_df[window_df['year'] == target_year][target].mean()

        X_seq.append(feature_seq)
        y_seq.append(target_data)
        years.append(target_year)

    return np.array(X_seq), np.array(y_seq), years

def compute_accuracy_metrics(model, X_test, y_test, scaler_y):
    # Predict scaled
    y_pred_scaled = model.predict(X_test)

    # Inverse transform to original scale
    y_test_orig = scaler_y.inverse_transform(y_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

    # Compute metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)

    return mse, mae, r2


# Sample output for structure validation
df_sample = pd.DataFrame({
    'year': np.tile(np.arange(2010, 2025), 2),
    'displ': np.random.rand(30),
    'city08': np.random.rand(30),
    'highway08': np.random.rand(30),
    'comb08': np.random.rand(30),
    'co2TailpipeGkm_converted': np.random.rand(30) * 100
})

def train_cnn_lstm_model(X_seq, y_seq, seq_len=5):
    # Flatten & scale X
    n_samples, _, n_feats = X_seq.shape
    flat_X = X_seq.reshape(-1, n_feats)
    scaler_X = MinMaxScaler().fit(flat_X)
    scaled_flat_X = scaler_X.transform(flat_X)
    X_scaled = scaled_flat_X.reshape(n_samples, seq_len, n_feats)

    # Scale y
    scaler_y = MinMaxScaler().fit(y_seq.reshape(-1,1))
    y_scaled = scaler_y.transform(y_seq.reshape(-1,1))

    # save scalers
    st.session_state.scaler_X_cnn = scaler_X
    st.session_state.scaler_y_cnn = scaler_y

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42)

    inp = Input((seq_len, n_feats))
    x = Conv1D(64, 1, activation="relu")(inp)
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(1)(x)

    model = Model(inp, out)
    model.compile("adam", "mse")
    es = EarlyStopping("val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=32, callbacks=[es], verbose=0)

    # Predictions & accuracy metrics
    preds = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, scaler_X, scaler_y, mse, mae, r2



def train_transformer_model(X_seq, y_seq, seq_len=5):
    # Flatten & scale X
    n_samples, _, n_feats = X_seq.shape
    flat_X = X_seq.reshape(-1, n_feats)
    scaler_X = MinMaxScaler().fit(flat_X)
    X_scaled = scaler_X.transform(flat_X).reshape(n_samples, seq_len, n_feats)

    # Scale y
    scaler_y = MinMaxScaler().fit(y_seq.reshape(-1,1))
    y_scaled = scaler_y.transform(y_seq.reshape(-1,1))

    # Store scalers
    st.session_state.scaler_X_transformer = scaler_X
    st.session_state.scaler_y_transformer = scaler_y

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # Build model
    inp = Input((seq_len, n_feats))
    x = Dense(64, activation="relu")(inp)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    out = Dense(1)(x)

    model = Model(inp, out)
    model.compile("adam", "mse")
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=32,
              callbacks=[EarlyStopping("val_loss", patience=10, restore_best_weights=True)],
              verbose=0)

    # ➕ Compute and return accuracy metrics
    mse, mae, r2 = compute_accuracy_metrics(model, X_test, y_test, scaler_y)

    return model, scaler_X, scaler_y, mse, mae, r2








def forecast_co2_emissions(df):
    st.header("CO₂ Emissions Forecast for Next 5 Years 📈")

    # build 5-year sliding windows
    seq_len = 5
    X_seq, y_seq, _ = prepare_sequence_data(df, seq_len)
    model_type = st.sidebar.radio("Select Model for Forecasting", ("Transformer", "CNN-LSTM"))

    if st.button("Forecast"):
        if model_type == "Transformer":
            if st.session_state.transformer_model is None:
                m, sx, sy, mse, mae, r2 = train_transformer_model(X_seq, y_seq, seq_len)
                st.session_state.transformer_model = m
                st.session_state.scaler_X_transformer = sx
                st.session_state.scaler_y_transformer = sy
                st.session_state.metrics_transformer = (mse, mae, r2)
                model = m
            else:
                model = st.session_state.transformer_model
                scaler_X = st.session_state.scaler_X_transformer
                scaler_y = st.session_state.scaler_y_transformer
                mse, mae, r2 = st.session_state.metrics_transformer if hasattr(st.session_state, 'metrics_transformer') else (0, 0, 0)
        
                # Display metrics
                st.subheader("📊 Transformer Model Metrics")
                st.write(f"**MSE**: {mse:.2f}")
                st.write(f"**MAE**: {mae:.2f}")
                st.write(f"**R²**: {r2:.2f}")
        else:
            if st.session_state.cnn_lstm_model is None:
                m, sx, sy, mse, mae, r2 = train_cnn_lstm_model(X_seq, y_seq, seq_len)
                st.session_state.cnn_lstm_model = m
                st.session_state.scaler_X_cnn = sx
                st.session_state.scaler_y_cnn = sy
                st.session_state.metrics_cnn = (mse, mae, r2)
                model = m
                scaler_X = sx
                scaler_y = sy
            else:
                model = st.session_state.cnn_lstm_model
                scaler_X = st.session_state.scaler_X_cnn
                scaler_y = st.session_state.scaler_y_cnn  
                mse, mae, r2 = st.session_state.metrics_cnn if hasattr(st.session_state, 'metrics_cnn') else (0, 0, 0)


         
        
        

        st.write(f"### Currently Using Model: {model_type}")
        
        # find the last 5 real years
        years = sorted(df['year'].dropna().unique())
        last5 = years[-seq_len:]
        future_years = list(range(last5[-1]+1, last5[-1]+6))  # Get full 5 future years
        
        # Debug info
        st.write(f"Last known years: {last5}")
        st.write(f"Future years to predict: {future_years}")
        
        # Get features for prediction
        features = ['displ', 'city08', 'highway08', 'comb08']
        
        # Create input for first prediction (last 5 known years)
        last_window = np.array([
            df[df['year'] == y][features].mean().values for y in last5
        ])
        
        # Initialize predictions
        all_preds = []
        current_window = last_window.copy()
        
        # Make predictions one by one, using previous predictions in the sequence
        for i, next_year in enumerate(future_years):
            # Scale input
            flat_input = current_window.reshape(-1, 4)  # 4 features
            scaled_input = scaler_X.transform(flat_input)
            scaled_seq = scaled_input.reshape(1, seq_len, 4)
            
            # Predict
            pred_scaled = model.predict(scaled_seq)[0][0]
            
            # Scale back
            pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
            all_preds.append(pred)
            
            # Update window for next prediction if we're not at the end
            if i < len(future_years) - 1:
                # Shift window (remove first entry, add a synthetic entry for the predicted year)
                current_window = np.roll(current_window, -1, axis=0)
                # For the new entry, use average values from the dataset
                avg_features = df[features].mean().values
                current_window[-1] = avg_features
        
        # assemble & adjust + tax
        df_fc = pd.DataFrame({
            'Year': future_years,
            'Forecasted CO₂ Emissions (g/km)': all_preds
        })
        
        # Winter adjustment
        df_fc['Forecasted CO₂ Emissions (g/km)'] = [
            v*1.1 if is_winter_in_taiwan(datetime(y, 12, 1)) else v
            for y, v in zip(df_fc['Year'], df_fc['Forecasted CO₂ Emissions (g/km)'])
        ]
        
        # Add carbon tax calculation
        df_fc['Forecasted Carbon Tax (NT$)/km'] = df_fc['Forecasted CO₂ Emissions (g/km)'].apply(
            lambda x: calculate_carbon_tax(x, 300)
        )


        # Display results
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(df_fc)
        with c2:
                width=700, height=400
            )


def forecast_co2_emissions_personal(car_info):
    st.header("Forecasted CO₂ Emissions for the Next 5 Years (2025–2029) 🚗📈")

    # Use global dataset
    co2_data = data[['year', 'displ', 'city08', 'highway08', 'comb08', 'adjusted_co2TailpipeGkm_converted']].dropna()
    seq_len = 5
    features = ['displ', 'city08', 'highway08', 'comb08']
    target = 'adjusted_co2TailpipeGkm_converted'  # Make sure this matches the column in co2_data

    # Prepare sequence data
    X_seq, y_seq, _ = prepare_sequence_data(co2_data, seq_len, target=target)

    # Model selection
    model_type = st.sidebar.radio("Select Model for Personal Forecasting", ("Transformer", "CNN-LSTM"))

    if st.button("Forecast Personal CO₂"):
        # Train if needed
        if model_type == "Transformer":
            if st.session_state.transformer_model is None:
                model, scaler_X, scaler_y, mse, mae, r2 = train_transformer_model(X_seq, y_seq, seq_len)
                st.session_state.transformer_model = model
                st.session_state.scaler_X_transformer = scaler_X
                st.session_state.scaler_y_transformer = scaler_y
                st.session_state.metrics_transformer = (mse, mae, r2)
            else:
                model = st.session_state.transformer_model
                scaler_X = st.session_state.scaler_X_transformer
                scaler_y = st.session_state.scaler_y_transformer
                mse, mae, r2 = st.session_state.metrics_transformer if hasattr(st.session_state, 'metrics_transformer') else (0, 0, 0)
            
            # Display metrics
            st.subheader("📊 Transformer Model Metrics")
            st.write(f"**MSE**: {mse:.2f}")
            st.write(f"**MAE**: {mae:.2f}")
            st.write(f"**R²**: {r2:.2f}")
        else:
            if st.session_state.cnn_lstm_model is None:
                model, scaler_X, scaler_y, mse, mae, r2 = train_cnn_lstm_model(X_seq, y_seq, seq_len)
                st.session_state.cnn_lstm_model = model
                st.session_state.scaler_X_cnn = scaler_X
                st.session_state.scaler_y_cnn = scaler_y
                st.session_state.metrics_cnn = (mse, mae, r2)
            else:
                model = st.session_state.cnn_lstm_model
                scaler_X = st.session_state.scaler_X_cnn
                scaler_y = st.session_state.scaler_y_cnn
                mse, mae, r2 = st.session_state.metrics_cnn if hasattr(st.session_state, 'metrics_cnn') else (0, 0, 0)
            
            # Display metrics
            st.subheader("📊 CNN-LSTM Model Metrics")
            st.write(f"**MSE**: {mse:.2f}")
            st.write(f"**MAE**: {mae:.2f}")
            st.write(f"**R²**: {r2:.2f}")

        # Initial sequence using average of last 5 years
        last_years = sorted(co2_data['year'].unique())[-seq_len:]
        last_window = np.array([
            co2_data[co2_data['year'] == y][features].mean().values for y in last_years
        ])

        future_years = list(range(2025, 2030))
        predictions = []

        current_window = last_window.copy()
        for i, year in enumerate(future_years):
            # Prepare input
            flat_input = current_window.reshape(-1, len(features))
            scaled_input = scaler_X.transform(flat_input)
            reshaped = scaled_input.reshape(1, seq_len, len(features))

            # Predict and inverse scale
            pred_scaled = model.predict(reshaped)[0][0]
            pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]

            # Apply winter adjustment
            if is_winter_in_taiwan(datetime(year, 12, 1)):
                pred *= 1.1

            predictions.append(pred)

            if i < len(future_years) - 1:
                # Create a new row based on user's car + 1% yearly drift
                synthetic = np.array([
                    car_info['displ'] * (1 + 0.01 * (i + 1)),
                    car_info['city08'] * (1 + 0.01 * (i + 1)),
                    car_info['highway08'] * (1 + 0.01 * (i + 1)),
                    car_info['comb08'] * (1 + 0.01 * (i + 1))
                ])
                current_window = np.roll(current_window, -1, axis=0)
                current_window[-1] = synthetic

        # Build DataFrame
        forecast_df = pd.DataFrame({
            "Year": future_years,
            "Forecasted CO₂ Emissions (g/km)": predictions
        })
        forecast_df['Forecasted Carbon Tax (NT$)/km'] = forecast_df['Forecasted CO₂ Emissions (g/km)'].apply(
        st.line_chart(forecast_df.set_index('Year')['Forecasted CO₂ Emissions (g/km)'])
            lambda x: calculate_carbon_tax(x, 300)
        )

        # Display
        st.dataframe(forecast_df)
    st.header("Personal CO2 Calculator 🚗💨")
    
    # Load the dataset
    data = load_data(FILE_PATH)
    
    # Filter Options
    years = data['year'].dropna().unique()
    selected_year = st.selectbox("Select Year", options=sorted(years))
    
    # Filter makes based on selected year
    makes = data[data['year'] == selected_year]['make'].dropna().unique()
    selected_make = st.selectbox("Select Make", options=sorted(makes))
    
    # Filter models based on selected year and make
    models = data[(data['year'] == selected_year) & (data['make'] == selected_make)]['model'].dropna().unique()
    selected_model = st.selectbox("Select Model", options=sorted(models))
    
    # Filter displacements based on selected year, make, and model
    displacements = data[(data['year'] == selected_year) & (data['make'] == selected_make) & (data['model'] == selected_model)]['displ'].dropna().unique()
    selected_displ = st.selectbox("Select Engine Size (L)", options=sorted(displacements))
    
    # Filter drive types based on selected year, make, model, and displacement
    drives = data[(data['year'] == selected_year) & (data['make'] == selected_make) & (data['model'] == selected_model) & (data['displ'] == selected_displ)]['drive'].dropna().unique()
    selected_drive = st.selectbox("Select Drive Type", options=sorted(drives))
    
    # Filter Data Based on Selection
    filtered_car = data[
        (data['year'] == selected_year) &
        (data['make'] == selected_make) &
        (data['model'] == selected_model) &
        (data['displ'] == selected_displ) &
        (data['drive'] == selected_drive)
    ]
    
    if not filtered_car.empty:
        car_info = filtered_car.iloc[0]
        st.subheader("Selected Car Information 🚗")
        st.write(f"**Year**: {car_info['year']}")
        st.write(f"**Make**: {car_info['make']}")
        st.write(f"**Model**: {car_info['model']}")
        st.write(f"**Engine Size (L)**: {car_info['displ']}")
        st.write(f"**Drive Type**: {car_info['drive']}")
        st.write(f"**CO₂ Emissions (g/km)**: {car_info['co2TailpipeGkm_converted']}")
        st.write(f"**Vehicle Class**: {car_info['VClass']}")
        st.write(f"**Transmission**: {car_info['trany']}")
        st.write(f"**Fuel Type**: {car_info['fuelType']}")
        
        # Forecast CO2 Emissions
        forecast_co2_emissions_personal(car_info)
    else:
        st.warning("⚠️ No car found with the selected criteria. Please try different options.")

if __name__ == "__main__":
    main()
