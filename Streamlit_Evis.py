import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Dropout, Input, LSTM, Flatten, Conv1D, GRU
import plotly.express as px
from datetime import datetime
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, NumeralTickFormatter
import plotly.graph_objects as go
import psutil
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers
import joblib
from tensorflow.keras.models import load_model
import os
import json

# Set page layout to centered
st.set_page_config(
    page_title="CO2 Emissions Analytics App",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Add these constants at the top of your file
models_dir = "trained_models"
MODEL_PATHS = {
    'rnn': {
        'weights': os.path.join(models_dir, 'rnn_model.weights.h5'),
        'config': os.path.join(models_dir, 'rnn_model_config.json')
    },
    'cnn_lstm': {
        'weights': os.path.join(models_dir, 'cnn_lstm_model.weights.h5'),
        'config': os.path.join(models_dir, 'cnn_lstm_model_config.json')
    },
    'gru': {
        'weights': os.path.join(models_dir, 'gru_model.weights.h5'),
        'config': os.path.join(models_dir, 'gru_model_config.json')
    }
}
SCALER_PATH = os.path.join(models_dir, 'scaler.pkl')
SEQ_LENGTH = 5

def create_model(config):
    """Create model from configuration."""
    try:
        inp = Input(config['input_shape'])
        
        if config['model_type'] == 'rnn':
            x = LSTM(64, return_sequences=True)(inp)
            x = LayerNormalization()(x)
            x = Dropout(0.2)(x)
            x = LSTM(32)(x)
        elif config['model_type'] == 'cnn_lstm':
            x = Conv1D(64, 2, padding='same', activation='relu')(inp)
            x = LayerNormalization()(x)
            x = LSTM(64, return_sequences=True)(x)
            x = LayerNormalization()(x)
            x = LSTM(32)(x)
        else:  # gru
            x = GRU(64, return_sequences=True)(inp)
            x = LayerNormalization()(x)
            x = Dropout(0.2)(x)
            x = GRU(32)(x)
        
        x = LayerNormalization()(x)
        out = Dense(1)(x)
        
        model = Model(inp, out)
        model.compile(**config['compile_args'])
        
        return model
    
    except Exception as e:
        st.error(f"Error creating model: {str(e)}")
        return None

def load_models_and_scaler():
    """Load trained models and scaler"""
    models = {}
    try:
        for model_type in MODEL_PATHS:
            config_path = MODEL_PATHS[model_type]['config']
            weights_path = MODEL_PATHS[model_type]['weights']
            
            if os.path.exists(config_path) and os.path.exists(weights_path):
                # Load configuration
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Create and compile model
                model = create_model(config)
                if model is not None:
                    # Load weights
                    model.load_weights(weights_path)
                    models[model_type] = model
            else:
                st.warning(f"Model files for {model_type} not found")
        
        if os.path.exists(SCALER_PATH):
            scalers = joblib.load(SCALER_PATH)
            # Extract feature and target scalers
            feature_scaler = scalers['feature_scaler']
            target_scaler = scalers['target_scaler']
        else:
            st.warning(f"Scaler not found at {SCALER_PATH}")
            feature_scaler, target_scaler = None, None
            
        return models, (feature_scaler, target_scaler)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, (None, None)

def prepare_sequence_data(df, seq_len=SEQ_LENGTH, target='co2TailpipeGkm_converted'):
    """Prepare sequence data for model input"""
    # Sort data by year
    df = df.sort_values('year')
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(df) - seq_len):
        seq = df[target].iloc[i:i+seq_len].values
        target_val = df[target].iloc[i+seq_len]
        sequences.append(seq)
        targets.append(target_val)
    
    return np.array(sequences), np.array(targets)

def make_prediction(model, sequence, scaler):
    """Make a prediction using the specified model"""
    try:
        # Reshape sequence based on model type
        if 'cnn_lstm' in str(model.name).lower():
            sequence = sequence.reshape((1, SEQ_LENGTH, 1))
        else:
            sequence = sequence.reshape((1, SEQ_LENGTH, 1))
        
        # Make prediction
        scaled_pred = model.predict(sequence, verbose=0)
        prediction = scaler.inverse_transform(scaled_pred.reshape(-1, 1))
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def calculate_carbon_tax(co2_emissions, tax_rate_per_ton=300):
    """Calculate carbon tax based on CO2 emissions (g/km)"""
    # Convert g/km to tons/km
    tons_per_km = co2_emissions / 1000000  # Convert g to tons
    # Calculate tax per km
    tax_per_km = tons_per_km * tax_rate_per_ton
    return tax_per_km

def forecast_co2_emissions(df):
    """Forecast CO2 emissions using selected model"""
    st.header("CO2 Emissions Forecast")
    
    # Sidebar for model selection
    st.sidebar.header("Forecast Settings")
    model_options = {
        'RNN': 'rnn',
        'CNN-LSTM': 'cnn_lstm',
        'GRU': 'gru'
    }
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys())
    )
    
    # Tax rate input (NT$)
    tax_rate = st.sidebar.number_input(
        "Carbon Tax Rate (NT$ per ton)",
        min_value=1,
        value=300,
        help="Enter the carbon tax rate per ton of CO2 in NT$"
    )
    
    # Forecast button
    if st.sidebar.button("Generate Forecast"):
        try:
            # Load models and scalers
            models, (feature_scaler, target_scaler) = load_models_and_scaler()
            
            if not models or feature_scaler is None or target_scaler is None:
                st.error("Unable to load models or scalers. Please ensure models are trained and saved correctly.")
                return
            
            # Get the selected model
            model_type = model_options[selected_model]
            model = models[model_type]
            
            # Get the most recent 5 years of data
            recent_years_data = df.sort_values('year').groupby('year').agg({
                'displ': 'mean',
                'cylinders': 'mean',
                'city08': 'mean',
                'highway08': 'mean',
                'comb08': 'mean',
                'co2TailpipeGkm_converted': 'mean'
            }).tail(5)
            
            # Prepare features
            feature_cols = ['displ', 'cylinders', 'city08', 'highway08', 'comb08']
            recent_features = recent_years_data[feature_cols].values
            
            # Scale features
            scaled_features = feature_scaler.transform(recent_features)
            
            # Use the last 3 years as the input sequence for forecasting
            current_sequence = scaled_features[-3:].copy().reshape(1, 3, len(feature_cols))
            
            # Get the last actual CO₂ value (for reference)
            last_known_co2 = recent_years_data['co2TailpipeGkm_converted'].iloc[-1]
            
            # Generate 5-year forecast
            forecast_years = range(2025, 2030)
            forecast_values = []
            
            for year_idx, year in enumerate(forecast_years):
                # Predict next CO₂ value
                pred_scaled = model.predict(current_sequence, verbose=0)
                pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

                # For the first forecasted year, force the prediction to start from the last known value
                if year_idx == 0:
                    pred = last_known_co2

                # Add a small, consistent trend (e.g., -1% per year)
                pred *= (0.99 ** year_idx)

                # Ensure prediction is within reasonable bounds
                min_bound = last_known_co2 * 0.7
                max_bound = last_known_co2 * 1.1
                pred = np.clip(pred, min_bound, max_bound)

                forecast_values.append(pred)

                # Generate next synthetic features by carrying forward the last feature row with a small trend
                new_features = current_sequence[0, -1].copy()
                new_features = new_features * (0.99 ** (year_idx + 1))  # Apply a -1% trend to all features

                # Roll the sequence and update
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1] = new_features
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Year': forecast_years,
                'Forecasted CO₂ Emissions (g/km)': forecast_values
            })
            
            # Calculate carbon tax
            forecast_df['Carbon Tax (NT$/km)'] = forecast_df['Forecasted CO₂ Emissions (g/km)'].apply(
                lambda x: calculate_carbon_tax(x, tax_rate)
            )
            
            # Display results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.subheader("Forecast Results")
                # Round the values to 2 decimal places for better display
                forecast_df['Forecasted CO₂ Emissions (g/km)'] = forecast_df['Forecasted CO₂ Emissions (g/km)'].round(2)
                forecast_df['Carbon Tax (NT$/km)'] = forecast_df['Carbon Tax (NT$/km)'].round(4)
                st.dataframe(forecast_df)
                
                # Add download button
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast Data",
                    data=csv,
                    file_name=f"co2_forecast_{selected_model}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.subheader("Forecast Visualization")
                
                # Create visualization
                fig = go.Figure()
                
                # Plot historical data
                historical_years = recent_years_data.index
                historical_values = recent_years_data['co2TailpipeGkm_converted']
                
                fig.add_trace(go.Scatter(
                    x=historical_years,
                    y=historical_values,
                    name='Historical Data',
                    line=dict(color='blue')
                ))
                
                # Plot forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['Year'],
                    y=forecast_df['Forecasted CO₂ Emissions (g/km)'],
                    name=f'{selected_model} Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f'CO2 Emissions Forecast ({selected_model})',
                    xaxis_title='Year',
                    yaxis_title='CO2 Emissions (g/km)',
                    template='plotly_dark',
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in forecasting: {str(e)}")

# Helper function: Is it winter in Taiwan?
def is_winter_in_taiwan(modified_date):
    return modified_date.month in [12, 1, 2]

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
FILE_PATH = './datasets/Vehicles_10-24.csv'

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
        try:
            scatter_chart = px.scatter(filtered_df, x='displ', y='co2TailpipeGkm_converted',
                                    title='Engine Size vs CO₂ Emissions',
                                    labels={'displ': 'Engine Displacement (L)', 'co2TailpipeGkm_converted': 'CO₂ Emissions (g/km)'})
        except ImportError:
            st.warning("Statsmodels package not found. Trendline functionality is disabled.")
            scatter_chart = px.scatter(filtered_df, x='displ', y='co2TailpipeGkm_converted',
                                    title='Engine Size vs CO₂ Emissions',
                                    labels={'displ': 'Engine Displacement (L)', 'co2TailpipeGkm_converted': 'CO₂ Emissions (g/km)'})
        st.plotly_chart(scatter_chart)

    # Forecasting
    forecast_co2_emissions(data)

    







# Sample DataFrame structure for testing
def prepare_sequence_data(df, seq_len=5, target='co2TailpipeGkm_converted'):
    # Sort by year and remove outliers
    df = df.sort_values('year')
    
    # Calculate z-scores for numerical columns
    numeric_cols = ['displ', 'city08', 'highway08', 'comb08', 'co2TailpipeGkm_converted']
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df = df[z_scores < 3]  # Remove outliers beyond 3 standard deviations
    
    # Feature engineering
    df['fuel_efficiency'] = (df['city08'] + df['highway08']) / 2
    df['engine_efficiency'] = df['comb08'] / df['displ']
    df['emission_per_displacement'] = df['co2TailpipeGkm_converted'] / df['displ']
    
    # Group by year with enhanced statistics
    agg_dict = {
        'displ': ['mean', 'std', 'min', 'max'],
        'city08': ['mean', 'std', 'min', 'max'],
        'highway08': ['mean', 'std', 'min', 'max'],
        'comb08': ['mean', 'std', 'min', 'max'],
        'co2TailpipeGkm_converted': ['mean', 'std', 'min', 'max', 'count'],
        'fuel_efficiency': ['mean', 'std'],
        'engine_efficiency': ['mean', 'std'],
        'emission_per_displacement': ['mean', 'std']
    }
    
    yearly_stats = df.groupby('year').agg(agg_dict).reset_index()
    yearly_stats.columns = ['year'] + [f"{col[0]}_{col[1]}" for col in yearly_stats.columns[1:]]
    
    # Add year-over-year changes
    for col in yearly_stats.columns[1:]:
        yearly_stats[f'{col}_yoy'] = yearly_stats[col].pct_change()
    
    # Fill NaN values with 0 for year-over-year changes
    yearly_stats = yearly_stats.fillna(0)
    
    # Create sequences with rolling statistics
    X_seq, y_seq, years = [], [], []
    
    for i in range(len(yearly_stats) - seq_len):
        # Get sequence window
        window = yearly_stats.iloc[i:i+seq_len]
        target_year = yearly_stats.iloc[i+seq_len]
        
        # Create feature vector including all statistics
        features = []
        for _, year_data in window.iterrows():
            # Exclude 'year' column and create feature vector
            year_features = year_data.drop('year').values
            features.append(year_features)
        
        X_seq.append(features)
        y_seq.append(target_year['co2TailpipeGkm_converted_mean'])
        years.append(target_year['year'])

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


def train_rnn_model(X_seq, y_seq, seq_len=5):
    n_samples, _, n_feats = X_seq.shape
    
    # Robust scaling with clipping
    flat_X = X_seq.reshape(-1, n_feats)
    scaler_X = MinMaxScaler(feature_range=(-1, 1)).fit(flat_X)
    X_scaled = scaler_X.transform(flat_X).reshape(n_samples, seq_len, n_feats)

    # Robust scaling for target
    scaler_y = MinMaxScaler(feature_range=(-1, 1)).fit(y_seq.reshape(-1,1))
    y_scaled = scaler_y.transform(y_seq.reshape(-1,1))

    # Store scalers
    st.session_state.scaler_X_transformer = scaler_X
    st.session_state.scaler_y_transformer = scaler_y

    # Time-based split
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # Build enhanced model
    inp = Input((seq_len, n_feats))
    
    # Multi-scale processing
    # Scale 1: Process entire sequence
    x1 = tf.keras.layers.Bidirectional(LSTM(128, return_sequences=True))(inp)
    x1 = LayerNormalization()(x1)
    x1 = Dropout(0.2)(x1)
    
    # Scale 2: Process with CNN for local patterns
    x2 = Conv1D(64, 2, padding='same', activation='relu')(inp)
    x2 = LayerNormalization()(x2)
    x2 = Conv1D(64, 2, padding='same', activation='relu')(x2)
    x2 = LayerNormalization()(x2)
    
    # Merge scales
    x = tf.keras.layers.Concatenate()([x1, x2])
    
    # Process merged features
    x = tf.keras.layers.Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Self-attention mechanism
    attention = MultiHeadAttention(num_heads=8, key_dim=32)(x, x)
    x = LayerNormalization()(attention + x)
    
    # Final sequence processing
    x = tf.keras.layers.Bidirectional(LSTM(32))(x)
    x = LayerNormalization()(x)
    
    # Dense processing with residual connections
    dense1 = Dense(64, activation='relu')(x)
    x = LayerNormalization()(dense1)
    x = Dropout(0.1)(x)
    
    dense2 = Dense(32, activation='relu')(x)
    x = LayerNormalization()(dense2)
    x = Dropout(0.1)(x)
    
    # Merge with original features
    x = Dense(32, activation='relu')(x)
    x = LayerNormalization()(x)
    x = layers.Add()([x, Dense(32)(dense1)])
    
    # Output with custom activation
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='linear')(x)
    
    model = Model(inp, out)
    
    # Advanced learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=50,
        t_mul=2.0,
        m_mul=0.9,
        alpha=1e-5
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=0.001
    )
    
    # Custom loss function combining MSE and MAE
    def custom_loss(y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        return 0.7 * mse + 0.3 * mae
    
    model.compile(
        optimizer=optimizer,
        loss=custom_loss,
        metrics=['mae', 'mse']
    )
    
    # Enhanced callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        min_delta=1e-5
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=1e-6
    )
    
    # Train with cross-validation
    n_splits = 5
    val_scores = []
    
    for fold in range(n_splits):
        # Create fold indices
        val_start = len(X_train) * fold // n_splits
        val_end = len(X_train) * (fold + 1) // n_splits
        
        # Split data for this fold
        X_train_fold = np.concatenate([X_train[:val_start], X_train[val_end:]])
        y_train_fold = np.concatenate([y_train[:val_start], y_train[val_end:]])
        X_val_fold = X_train[val_start:val_end]
        y_val_fold = y_train[val_start:val_end]
        
        # Train on this fold
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=400,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        val_scores.append(min(history.history['val_loss']))
    
    # Final training on all training data
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=400,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    
    # Compute metrics
    y_pred = model.predict(X_test)
    y_test_orig = scaler_y.inverse_transform(y_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)
    
    # Print detailed metrics
    st.write("### Model Training Summary")
    st.write(f"Average validation score: {np.mean(val_scores):.4f} (±{np.std(val_scores):.4f})")
    st.write(f"Final MSE: {mse:.4f}")
    st.write(f"Final MAE: {mae:.4f}")
    st.write(f"Final R²: {r2:.4f}")
    
    # Plot predictions vs actual
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(y=y_test_orig.flatten(), name='Actual', line=dict(color='blue')))
    fig_pred.add_trace(go.Scatter(y=y_pred_orig.flatten(), name='Predicted', line=dict(color='red')))
    fig_pred.update_layout(
        title='Predictions vs Actual Values',
        xaxis_title='Sample',
        yaxis_title='CO2 Emissions',
        template='plotly_dark',
        showlegend=True
    )
    st.plotly_chart(fig_pred)
    
    # Plot error distribution
    errors = y_test_orig.flatten() - y_pred_orig.flatten()
    fig_error = go.Figure()
    fig_error.add_trace(go.Histogram(x=errors, nbinsx=30, name='Error Distribution'))
    fig_error.update_layout(
        title='Prediction Error Distribution',
        xaxis_title='Error',
        yaxis_title='Count',
        template='plotly_dark'
    )
    st.plotly_chart(fig_error)
    
    return model, scaler_X, scaler_y, mse, mae, r2

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

    # Split with more recent data in test set
    split_idx = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

    # Enhanced CNN-LSTM architecture
    inp = Input((seq_len, n_feats))
    
    # CNN layers for feature extraction
    x = Conv1D(128, 2, padding='same', activation='relu')(inp)
    x = LayerNormalization()(x)
    x = Conv1D(64, 2, padding='same', activation='relu')(x)
    x = LayerNormalization()(x)
    
    # Add residual connection
    res = Conv1D(64, 1, padding='same')(inp)
    x = layers.Add()([x, res])
    x = Dropout(0.2)(x)
    
    # LSTM layers for temporal processing
    x = LSTM(128, return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(64, return_sequences=True)(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = LSTM(32)(x)
    x = LayerNormalization()(x)
    
    # Dense layers with residual connections
    dense1 = Dense(32, activation="relu")(x)
    x = LayerNormalization()(dense1)
    x = Dropout(0.1)(x)
    
    dense2 = Dense(16, activation="relu")(x)
    x = LayerNormalization()(dense2)
    x = Dropout(0.1)(x)
    
    # Final residual connection
    x = Dense(32, activation="relu")(x)
    x = LayerNormalization()(x)
    x = layers.Add()([x, dense1])
    
    # Output layer
    out = Dense(1)(x)

    model = Model(inp, out)
    
    # Learning rate scheduling
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile with Huber loss
    model.compile(
        optimizer=optimizer,
        loss="huber",
        metrics=["mae", "mse"]
    )

    # Enhanced early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        min_delta=1e-4
    )

    # Train with increased epochs and batch size
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0
    )

    # Get predictions and compute metrics
    y_pred = model.predict(X_test)
    
    # Transform back to original scale for metrics
    y_test_orig = scaler_y.inverse_transform(y_test)
    y_pred_orig = scaler_y.inverse_transform(y_pred)
    
    # Compute metrics
    mse = mean_squared_error(y_test_orig, y_pred_orig)
    mae = mean_absolute_error(y_test_orig, y_pred_orig)
    r2 = r2_score(y_test_orig, y_pred_orig)

    # Print training summary
    st.write("Model Training Summary:")
    st.write(f"Final training loss: {history.history['loss'][-1]:.4f}")
    st.write(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    # Plot training history
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
    fig.update_layout(
        title='CNN-LSTM Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        template='plotly_dark'
    )
    st.plotly_chart(fig)

    return model, scaler_X, scaler_y, mse, mae, r2






def forecast_co2_emissions_personal(car_info):
    st.header("Forecasted CO₂ Emissions for the Next 5 Years (2025–2029) 🚗📈")

    # Use global dataset
    co2_data = data[['year', 'displ', 'cylinders', 'city08', 'highway08', 'comb08', 'co2TailpipeGkm_converted']].dropna()
    seq_len = 3
    feature_cols = ['displ', 'cylinders', 'city08', 'highway08', 'comb08']
    target_col = 'co2TailpipeGkm_converted'

    # Sidebar for model selection and tax rate
    st.sidebar.header("Personal Forecast Settings")
    model_options = {
        'RNN': 'rnn',
        'CNN-LSTM': 'cnn_lstm',
        'GRU': 'gru'
    }
    selected_model = st.sidebar.selectbox(
        "Select Model for Personal Forecasting",
        list(model_options.keys())
    )
    tax_rate = st.sidebar.number_input(
        "Carbon Tax Rate (NT$ per ton)",
        min_value=1,
        value=300,
        help="Enter the carbon tax rate per ton of CO2 in NT$"
    )

    if st.button("Forecast Personal CO₂"):
        # Load models and scalers
        models, (feature_scaler, target_scaler) = load_models_and_scaler()
        if not models or feature_scaler is None or target_scaler is None:
            st.error("Unable to load models or scalers. Please ensure models are trained and saved correctly.")
            return

        model_type = model_options[selected_model]
        model = models[model_type]

        # Prepare historical data for visualization
        recent_years_data = co2_data.groupby('year').agg({
            'displ': 'mean',
            'cylinders': 'mean',
            'city08': 'mean',
            'highway08': 'mean',
            'comb08': 'mean',
            'co2TailpipeGkm_converted': 'mean'
        }).sort_index()
        last_years = recent_years_data.tail(seq_len)
        last_features = last_years[feature_cols].values
        scaled_features = feature_scaler.transform(last_features)
        current_sequence = scaled_features.copy().reshape(1, seq_len, len(feature_cols))

        # Forecast for the next 5 years
        forecast_years = list(range(2025, 2030))
        forecast_values = []
        for i, year in enumerate(forecast_years):
            pred_scaled = model.predict(current_sequence, verbose=0)
            pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            # Optionally, apply a small trend or adjustment here if desired

            # Apply winter adjustment if needed
            if is_winter_in_taiwan(datetime(year, 12, 1)):
                pred *= 1.1

            forecast_values.append(pred)

            # Generate next synthetic features based on user's car (with a small drift)
            new_features = np.array([
                car_info['displ'] * (1 + 0.01 * (i + 1)),
                car_info['cylinders'] if 'cylinders' in car_info else 4,
                car_info['city08'] * (1 + 0.01 * (i + 1)),
                car_info['highway08'] * (1 + 0.01 * (i + 1)),
                car_info['comb08'] * (1 + 0.01 * (i + 1))
            ])
            new_features_scaled = feature_scaler.transform(new_features.reshape(1, -1))
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = new_features_scaled

        # Build DataFrame
        forecast_df = pd.DataFrame({
            "Year": forecast_years,
            "Forecasted CO₂ Emissions (g/km)": forecast_values
        })
        forecast_df['Carbon Tax (NT$/km)'] = forecast_df['Forecasted CO₂ Emissions (g/km)'].apply(
            lambda x: calculate_carbon_tax(x, tax_rate)
        )

        # Display results
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Personal Forecast Results")
            forecast_df['Forecasted CO₂ Emissions (g/km)'] = forecast_df['Forecasted CO₂ Emissions (g/km)'].round(2)
            forecast_df['Carbon Tax (NT$/km)'] = forecast_df['Carbon Tax (NT$/km)'].round(4)
            st.dataframe(forecast_df)
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="Download Personal Forecast Data",
                data=csv,
                file_name=f"personal_co2_forecast_{selected_model}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        with col2:
            st.subheader("Personal Forecast Visualization")
            fig = go.Figure()
            # Plot historical data
            fig.add_trace(go.Scatter(
                x=recent_years_data.index,
                y=recent_years_data['co2TailpipeGkm_converted'],
                name='Historical Data',
                line=dict(color='blue')
            ))
            # Plot forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['Year'],
                y=forecast_df['Forecasted CO₂ Emissions (g/km)'],
                name=f'{selected_model} Forecast',
                line=dict(color='red', dash='dash')
            ))
            fig.update_layout(
                title=f'Personal CO2 Emissions Forecast ({selected_model})',
                xaxis_title='Year',
                yaxis_title='CO2 Emissions (g/km)',
                template='plotly_dark',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)


def upload_predict_page():
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
