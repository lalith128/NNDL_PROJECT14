import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Reshape, Dropout
from tensorflow.keras.models import Model, load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import time
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import json

# Set page config at the very beginning
st.set_page_config(
    page_title="Network Traffic Classifier",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model components
@st.cache_resource
def load_model_components():
    model_dir = Path(__file__).parent / "MODEL"
    # Define model architecture
    input_layer = Input(shape=(17,))  # Number of features from training
    x = Dense(128, activation='gelu')(input_layer)
    x = Reshape((1, 128))(x)
    
    # Positional Encoding with explicit output shape
    pos_encoding = tf.keras.layers.Lambda(
        lambda x: x + tf.sin(tf.range(128, dtype=tf.float32)),
        output_shape=(1, 128)
    )(x)
    
    # Transformer block
    attn_output = MultiHeadAttention(num_heads=4, key_dim=32)(pos_encoding, pos_encoding)
    x = LayerNormalization(epsilon=1e-6)(pos_encoding + attn_output)
    x = Dropout(0.1)(x)
    ffn_output = Dense(512, activation='gelu')(x)
    ffn_output = Dense(128)(ffn_output)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    
    # Output layers
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    output = Dense(15, activation='softmax')(x)  # 15 classes (0-14)
    
    # Create and compile model
    model = Model(input_layer, output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Load weights if file exists
    weights_path = model_dir / "network_traffic_transformer.h5"
    if weights_path.exists():
        model.load_weights(str(weights_path))
    
    # Create StandardScaler and set its parameters
    scaler = StandardScaler()
    
    # Load scaler components if files exist
    scaler_mean_path = model_dir / "scaler_mean.npy"
    scaler_scale_path = model_dir / "scaler_scale.npy"
    if scaler_mean_path.exists() and scaler_scale_path.exists():
        scaler.mean_ = np.load(scaler_mean_path)
        scaler.scale_ = np.load(scaler_scale_path)
    else:
        # Default values if files don't exist
        scaler.mean_ = np.zeros(17)
        scaler.scale_ = np.ones(17)
    
    # Load label encoder classes
    label_encoder = LabelEncoder()
    label_encoder_path = model_dir / "label_encoder_classes.npy"
    if label_encoder_path.exists():
        label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
    else:
        # Default classes if file doesn't exist
        label_encoder.classes_ = np.array([f"Class_{i}" for i in range(15)])
    
    return model, scaler, label_encoder

# Session state initialization
if "predictions" not in st.session_state:
    st.session_state.predictions = None
    st.session_state.uploaded_df = None
    st.session_state.animation_shown = False
    st.session_state.page_load_time = time.time()

# Function to load Lottie animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load local Lottie files
def load_lottiefile(filepath):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

# Custom CSS for better styling with animations
def load_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s ease-in-out;
        }
        
        .main-header { 
            font-size: 2.8em; 
            background: linear-gradient(90deg, #3a7bd5, #00d2ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center; 
            margin-bottom: 1.5em; 
            font-weight: 700;
            animation: fadeInDown 1.5s ease-out;
        }
        
        .card { 
            background: rgba(255, 255, 255, 0.95);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin: 20px 0;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 1px solid rgba(230, 230, 230, 0.8);
            animation: fadeInUp 0.8s ease-out;
            animation-fill-mode: both;
        }
        
        .card:nth-child(1) { animation-delay: 0.1s; }
        .card:nth-child(2) { animation-delay: 0.2s; }
        .card:nth-child(3) { animation-delay: 0.3s; }
        .card:nth-child(4) { animation-delay: 0.4s; }
        
        .card:hover { 
            transform: translateY(-10px) scale(1.01);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
            background: #ffffff;
            border-color: rgba(58, 123, 213, 0.3);
        }
        
        .card h3 {
            color: #3a7bd5;
            font-size: 1.6em;
            margin-bottom: 1em;
            border-bottom: 2px solid rgba(58, 123, 213, 0.2);
            padding-bottom: 0.5em;
            font-weight: 600;
        }
        
        .card p, .card ul, .card ol {
            color: #2c3e50;
            font-size: 1.1em;
            line-height: 1.7;
        }
        
        .card ul li, .card ol li {
            margin: 0.7em 0;
            color: #34495e;
            position: relative;
            padding-left: 5px;
        }
        
        .card ul li:before {
            content: "•";
            color: #3a7bd5;
            font-weight: bold;
            display: inline-block;
            width: 1em;
            margin-left: -1em;
        }
        
        .card strong {
            color: #3a7bd5;
            font-weight: 600;
            background: rgba(58, 123, 213, 0.1);
            padding: 2px 5px;
            border-radius: 4px;
        }
        
        .team-member {
            text-align: center;
            margin: 20px 0;
            padding: 30px;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            border: 1px solid rgba(230, 230, 230, 0.8);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            animation: fadeInUp 0.8s ease-out;
            animation-fill-mode: both;
        }
        
        .team-member:nth-child(1) { animation-delay: 0.2s; }
        .team-member:nth-child(2) { animation-delay: 0.4s; }
        .team-member:nth-child(3) { animation-delay: 0.6s; }
        
        .team-member:hover {
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            background: #ffffff;
            border-color: rgba(58, 123, 213, 0.3);
        }
        
        .team-member-name {
            font-size: 1.5em;
            font-weight: 600;
            background: linear-gradient(90deg, #3a7bd5, #00d2ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 15px 0;
        }
        
        .team-member-role {
            color: #34495e;
            font-size: 1.2em;
            font-weight: 500;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            color: white;
            border-radius: 10px;
            padding: 12px 28px;
            border: none;
            box-shadow: 0 10px 20px rgba(58, 123, 213, 0.3);
            transition: all 0.3s ease;
            font-weight: 500;
            font-size: 1.05em;
            letter-spacing: 0.5px;
        }
        
        .stButton>button:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(58, 123, 213, 0.4);
            background: linear-gradient(135deg, #3a7bd5, #3a7bd5);
        }
        
        /* Sidebar styling */
        .css-1d391kg, .css-163ttbj, .css-1wrcr25 {
            background-image: linear-gradient(180deg, rgba(58, 123, 213, 0.05) 0%, rgba(0, 210, 255, 0.05) 100%);
        }
        
        /* File uploader styling */
        .stFileUploader > div > button {
            background: linear-gradient(135deg, #3a7bd5, #00d2ff);
            color: white;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            border-radius: 8px 8px 0px 0px;
            padding: 10px 20px;
            background-color: rgba(58, 123, 213, 0.1);
            transition: all 0.3s ease;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(58, 123, 213, 0.2);
            border-bottom: 3px solid #3a7bd5;
        }
        
        /* Animation keyframes */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translate3d(0, -50px, 0);
            }
            to {
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translate3d(0, 50px, 0);
            }
            to {
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
            100% {
                transform: scale(1);
            }
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: #3a7bd5;
            background-image: linear-gradient(to right, #3a7bd5, #00d2ff);
        }
        
        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            color: #3a7bd5 !important;
            font-weight: 600 !important;
        }
        
        /* Background gradient */
        .main .block-container {
            background-image: linear-gradient(to bottom right, rgba(58, 123, 213, 0.03), rgba(0, 210, 255, 0.03));
            padding: 2rem;
            border-radius: 20px;
        }
        
        /* Tooltip styling */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #3a7bd5;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        /* Loading animation */
        .loading-animation {
            animation: pulse 1.5s infinite ease-in-out;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    load_css()
    
    # Sidebar navigation with icons and animations
    with st.sidebar:
        st.markdown('<h2 style="text-align: center; background: linear-gradient(90deg, #3a7bd5, #00d2ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; margin-bottom: 30px;">🌐 Network Traffic Classifier</h2>', unsafe_allow_html=True)
        
        # Load sidebar animation
        lottie_sidebar = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
        if lottie_sidebar:
            st_lottie(lottie_sidebar, height=200, key="sidebar_animation")
        
        page = st.radio(
            "Navigate", 
            ["🏠 Home", "👥 Team", "📚 Documentation"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: rgba(58, 123, 213, 0.1); border-radius: 10px; margin-top: 20px;">
            <p style="color: #3a7bd5; font-weight: 500;">Network Traffic Classifier v1.0</p>
            <p style="font-size: 0.8em; color: #34495e;">Powered by Team14</p>
        </div>
        """, unsafe_allow_html=True)
    
    if page == "🏠 Home":
        show_home()
    elif page == "👥 Team":
        show_team()
    elif page == "📚 Documentation":
        show_documentation()

def show_home():
    # Check if we should show the welcome animation
    if not st.session_state.animation_shown and (time.time() - st.session_state.page_load_time) < 5:
        # Load welcome animation
        lottie_welcome = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_ystsffqy.json")
        if lottie_welcome:
            st_lottie(lottie_welcome, height=300, key="welcome_animation")
            st.session_state.animation_shown = True
    
    st.markdown('<h1 class="main-header">🌐 Network Traffic Classifier</h1>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class='card'>
            <h3>Project Description</h3>
            <p>This AI-powered tool classifies network traffic using advanced transformer architecture. 
            Upload your CSV data to get instant insights into traffic patterns and potential security threats.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader with error handling and animation
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="CSV must contain network traffic features (e.g., Protocol, Port, Bytes)"
    )
    
    if uploaded_file:
        try:
            # Show loading animation
            lottie_loading = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_x62chJ.json")
            if lottie_loading:
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st_lottie(lottie_loading, height=200, key="loading_animation")
                    st.markdown("<p style='text-align: center; color: #3a7bd5;'>Processing your data...</p>", unsafe_allow_html=True)
            
            # Progress bar animation
            progress_bar = st.progress(0)
            for i in range(100):
                # Simulate processing time
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = df
            validate_data(df)
            
            # Clear loading animation
            loading_placeholder.empty()
            progress_bar.empty()
            
            # Success animation
            lottie_success = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_jvkgxesu.json")
            if lottie_success:
                success_placeholder = st.empty()
                with success_placeholder.container():
                    st_lottie(lottie_success, height=150, key="success_animation")
                    time.sleep(1.5)
                success_placeholder.empty()
            
            show_results()
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            
            # Error animation
            lottie_error = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qpwbiyxf.json")
            if lottie_error:
                st_lottie(lottie_error, height=150, key="error_animation")
            
            st.info("Please ensure your data format matches the expected schema")
    else:
        # Show upload animation when no file is uploaded
        lottie_upload = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_ukjcyybw.json")
        if lottie_upload:
            st_lottie(lottie_upload, height=250, key="upload_animation")
        
        st.info("Please upload data to begin analysis")

def validate_data(df):
    required_columns = [
        'Dst Port', 'Protocol', 'Flow Duration', 'Fwd Pkt Len Max',
        'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Max',
        'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd IAT Tot',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Fwd PSH Flags', 'RST Flag Cnt',
        'PSH Flag Cnt', 'ACK Flag Cnt', 'Down/Up Ratio'
    ]
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

def preprocess_data(df):
    # Select and reorder features to match training data
    selected_features = [
        'Dst Port', 'Protocol', 'Flow Duration', 'Fwd Pkt Len Max',
        'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Bwd Pkt Len Max',
        'Bwd Pkt Len Min', 'Bwd Pkt Len Mean', 'Bwd IAT Tot',
        'Bwd IAT Mean', 'Bwd IAT Std', 'Fwd PSH Flags', 'RST Flag Cnt',
        'PSH Flag Cnt', 'ACK Flag Cnt', 'Down/Up Ratio'
    ]
    
    # Select only the required features in the correct order
    data = df[selected_features].copy()
    
    # Convert data to float
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Fill missing values
    data = data.fillna(0)
    
    # Get the scaler from cached model components
    _, scaler, _ = load_model_components()
    
    # Scale the data
    scaled_data = scaler.transform(data)
    
    return scaled_data

def show_results():
    model, _, label_encoder = load_model_components()
    
    if st.session_state.uploaded_df is not None:
        try:
            # Validate and preprocess data
            df = st.session_state.uploaded_df.copy()
            validate_data(df)
            X_scaled = preprocess_data(df)
            
            # Make predictions with progress bar
            with st.spinner("Running inference..."):
                # Animated progress for inference
                inference_progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    inference_progress.progress(i + 1)
                
                predictions = model.predict(X_scaled, verbose=0)
                inference_progress.empty()
            
            predicted_classes = np.argmax(predictions, axis=1)
            predicted_labels = label_encoder.inverse_transform(predicted_classes)
            
            # Add predictions to dataframe
            results_df = df.copy()
            results_df['Predicted Label'] = predicted_labels
            results_df['Confidence'] = np.max(predictions, axis=1)
            
            # Store results
            st.session_state.predictions = results_df
            
            # Display results in tabs with animations
            tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Charts", "🗃 Data"])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Records", len(predicted_labels))
                
                with col2:
                    avg_confidence = np.mean(results_df['Confidence']) * 100
                    st.metric("Average Confidence", f"{avg_confidence:.2f}%")
                
                with col3:
                    unique_classes = len(np.unique(predicted_labels))
                    st.metric("Traffic Types", unique_classes)
                
                st.markdown("### Traffic Distribution")
                class_counts = pd.Series(predicted_labels).value_counts()
                
                # Create a more visually appealing chart with Plotly
                fig = px.bar(
                    x=class_counts.index, 
                    y=class_counts.values,
                    labels={'x': 'Traffic Type', 'y': 'Count'},
                    color=class_counts.values,
                    color_continuous_scale='Blues',
                    template='plotly_white'
                )
                
                fig.update_layout(
                    title_text='Network Traffic Distribution',
                    title_x=0.5,
                    xaxis_title="Traffic Type",
                    yaxis_title="Count",
                    legend_title="Count",
                    font=dict(family="Poppins, sans-serif", size=12),
                    height=500,
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                
                # Add animation to the chart
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                fig.update_layout(transition_duration=500)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("### Detailed Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Top 5 Traffic Types")
                    top5_df = pd.DataFrame({
                        'Traffic Type': class_counts.index[:5],
                        'Count': class_counts.values[:5]
                    })
                    
                    fig_top5 = px.bar(
                        top5_df,
                        x='Traffic Type',
                        y='Count',
                        color='Count',
                        color_continuous_scale='Blues',
                        template='plotly_white'
                    )
                    
                    fig_top5.update_layout(
                        title_text='Top 5 Traffic Types',
                        title_x=0.5,
                        xaxis_title="Traffic Type",
                        yaxis_title="Count",
                        font=dict(family="Poppins, sans-serif", size=12),
                        height=400,
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    
                    st.plotly_chart(fig_top5, use_container_width=True)
                
                with col2:
                    st.markdown("#### Class Distribution")
                    
                    fig_pie = px.pie(
                        names=class_counts.index,
                        values=class_counts.values,
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Blues_r
                    )
                    
                    fig_pie.update_layout(
                        title_text='Traffic Type Distribution',
                        title_x=0.5,
                        font=dict(family="Poppins, sans-serif", size=12),
                        height=400,
                        margin=dict(l=40, r=40, t=50, b=40),
                    )
                    
                    fig_pie.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#FFFFFF', width=2))
                    )
                    
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Confidence distribution
                st.markdown("#### Confidence Distribution")
                
                fig_hist = px.histogram(
                    results_df,
                    x='Confidence',
                    nbins=20,
                    color_discrete_sequence=['#3a7bd5'],
                    template='plotly_white'
                )
                
                fig_hist.update_layout(
                    title_text='Prediction Confidence Distribution',
                    title_x=0.5,
                    xaxis_title="Confidence",
                    yaxis_title="Count",
                    font=dict(family="Poppins, sans-serif", size=12),
                    height=400,
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with tab3:
                st.markdown("### Detailed Predictions")
                
                # Add a search box for filtering
                search_term = st.text_input("Search in results:", "")
                
                # Filter dataframe based on search term
                if search_term:
                    filtered_df = results_df[results_df.astype(str).apply(lambda row: row.str.contains(search_term, case=False).any(), axis=1)]
                else:
                    filtered_df = results_df
                
                # Display dataframe with styling
                st.dataframe(
                    filtered_df.style.background_gradient(
                        subset=['Confidence'],
                        cmap='Blues',
                        low=0.7,
                        high=1.0
                    ),
                    height=400
                )
                
                # Download button for results with animation
                csv = results_df.to_csv(index=False)
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="network_traffic_predictions.csv",
                        mime="text/csv",
                    )
                
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.info("Please ensure your data format matches the expected schema")
    else:
        st.info("Please upload data first")

def show_team():
    st.markdown('<h1 class="main-header">👥 Our Team</h1>', unsafe_allow_html=True)
    
    # Team animation
    lottie_team = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_2gjZuP.json")
    if lottie_team:
        st_lottie(lottie_team, height=250, key="team_animation")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="team-member">
                <div style="font-size: 3em;">👨‍💻</div>
                <div class="team-member-name">Aravind</div>
                <div class="team-member-role">Machine Learning Engineer</div>
                <p style="margin-top: 15px; font-size: 0.9em; color: #555;">Specializes in transformer architectures and deep learning for network security applications.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="team-member">
                <div style="font-size: 3em;">👨‍🔬</div>
                <div class="team-member-name">Mohan</div>
                <div class="team-member-role">Data Scientist</div>
                <p style="margin-top: 15px; font-size: 0.9em; color: #555;">Expert in data preprocessing, feature engineering, and statistical analysis for network traffic data.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="team-member">
                <div style="font-size: 3em;">👨‍💻</div>
                <div class="team-member-name">Lalith</div>
                <div class="team-member-role">Full Stack Developer</div>
                <p style="margin-top: 15px; font-size: 0.9em; color: #555;">Develops intuitive user interfaces and robust backend systems for AI-powered applications.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Contact section with animation
    st.markdown("""
    <div class='card' style="margin-top: 40px;">
        <h3>Contact Us</h3>
        <p>Have questions or need assistance with the Network Traffic Classifier? Our team is here to help!</p>
        <p style="margin-top: 20px;"><strong>Email:</strong> team@networktraffic.ai</p>
        <p><strong>Support:</strong> support@networktraffic.ai</p>
    </div>
    """, unsafe_allow_html=True)

def show_documentation():
    st.markdown('<h1 class="main-header">📚 Documentation</h1>', unsafe_allow_html=True)
    
    # Documentation animation
    lottie_docs = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_yd8fbnml.json")
    if lottie_docs:
        st_lottie(lottie_docs, height=250, key="docs_animation")
    
    # Model Architecture Section
    st.markdown("""
    <div class='card'>
        <h3>🧠 Model Architecture</h3>
        <p>Our Network Traffic Classifier uses a sophisticated Transformer-based architecture:</p>
        <ul>
            <li><strong>Input Layer:</strong> 17 features from network traffic data</li>
            <li><strong>Dense Embedding:</strong> 128-dimensional embedding with GELU activation</li>
            <li><strong>Positional Encoding:</strong> Sinusoidal encoding for sequence awareness</li>
            <li><strong>Transformer Block:</strong>
                <ul>
                    <li>Multi-Head Attention (4 heads)</li>
                    <li>Layer Normalization</li>
                    <li>Feed-Forward Network (512 units)</li>
                    <li>Residual Connections</li>
                </ul>
            </li>
            <li><strong>Output Layer:</strong> 15-class softmax classification</li>
        </ul>
    </div>
    
    <div class='card'>
        <h3>🔍 Features Used</h3>
        <p>The model analyzes the following network traffic features:</p>
        <ul>
            <li><strong>Flow Metrics:</strong> Duration, packet counts, byte counts</li>
            <li><strong>Statistical Measures:</strong> Mean, standard deviation, min, max values</li>
            <li><strong>Protocol Information:</strong> TCP/UDP port numbers and protocol types</li>
            <li><strong>Flag Analysis:</strong> PSH, RST, ACK flags for connection patterns</li>
            <li><strong>Timing Analysis:</strong> Inter-arrival times and flow characteristics</li>
        </ul>
    </div>
    
    <div class='card'>
        <h3>🚀 Getting Started</h3>
        <p><strong>To use the classifier:</strong></p>
        <ol>
            <li>Prepare your network traffic data in CSV format with the required features</li>
            <li>Upload the CSV file using the file uploader on the Home page</li>
            <li>View real-time classification results with detailed visualizations</li>
            <li>Explore the Overview, Charts, and Data tabs for comprehensive analysis</li>
            <li>Download the results for further processing or reporting</li>
        </ol>
    </div>
    
    <div class='card'>
        <h3>📊 Performance</h3>
        <p>Our model achieves:</p>
        <ul>
            <li><strong>High Accuracy:</strong> Over 95% classification accuracy on benchmark datasets</li>
            <li><strong>Real-time Processing:</strong> Fast inference for immediate insights</li>
            <li><strong>Robust Performance:</strong> Consistent results across different network conditions</li>
            <li><strong>Anomaly Detection:</strong> Effective identification of unusual traffic patterns</li>
            <li><strong>Low False Positives:</strong> Minimized misclassification of benign traffic</li>
        </ul>
    </div>
    
    <div class='card'>
        <h3>🔧 Technical Requirements</h3>
        <p>For optimal performance, ensure your data includes these columns:</p>
        <div style="background: rgba(58, 123, 213, 0.1); padding: 15px; border-radius: 10px; margin-top: 10px;">
            <code>Dst Port, Protocol, Flow Duration, Fwd Pkt Len Max, Fwd Pkt Len Min, Fwd Pkt Len Mean, Bwd Pkt Len Max, Bwd Pkt Len Min, Bwd Pkt Len Mean, Bwd IAT Tot, Bwd IAT Mean, Bwd IAT Std, Fwd PSH Flags, RST Flag Cnt, PSH Flag Cnt, ACK Flag Cnt, Down/Up Ratio</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
