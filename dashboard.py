import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from streamlit_lottie import st_lottie
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config at the very beginning
st.set_page_config(
    page_title="Network Traffic Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "animation_shown" not in st.session_state:
    st.session_state.animation_shown = False
if "page_load_time" not in st.session_state:
    st.session_state.page_load_time = time.time()
if "notifications" not in st.session_state:
    st.session_state.notifications = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "dashboard"

# Function to load Lottie animations
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Custom CSS for better styling with animations
def load_css():
    # Determine theme based on dark mode setting
    if st.session_state.dark_mode:
        primary_color = "#00d2ff"
        secondary_color = "#3a7bd5"
        bg_color = "#121212"
        card_bg = "#1E1E1E"
        text_color = "#E0E0E0"
        card_border = "rgba(255, 255, 255, 0.1)"
        hover_bg = "#2A2A2A"
    else:
        primary_color = "#3a7bd5"
        secondary_color = "#00d2ff"
        bg_color = "#FFFFFF"
        card_bg = "rgba(255, 255, 255, 0.95)"
        text_color = "#2c3e50"
        card_border = "rgba(230, 230, 230, 0.8)"
        hover_bg = "#FFFFFF"
    
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        * {{
            font-family: 'Poppins', sans-serif;
            transition: all 0.3s ease-in-out;
        }}
        
        .main .block-container {{
            background-color: {bg_color};
            background-image: linear-gradient(to bottom right, 
                {primary_color}05, {secondary_color}05);
            padding: 2rem;
            border-radius: 20px;
        }}
        
        .main-header {{ 
            font-size: 2.8em; 
            background: linear-gradient(90deg, {primary_color}, {secondary_color});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center; 
            margin-bottom: 1.5em; 
            font-weight: 700;
            animation: fadeInDown 1.5s ease-out;
        }}
        
        .card {{ 
            background: {card_bg};
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            margin: 20px 0;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            border: 1px solid {card_border};
            animation: fadeInUp 0.8s ease-out;
            animation-fill-mode: both;
        }}
        
        .card:nth-child(1) {{ animation-delay: 0.1s; }}
        .card:nth-child(2) {{ animation-delay: 0.2s; }}
        .card:nth-child(3) {{ animation-delay: 0.3s; }}
        .card:nth-child(4) {{ animation-delay: 0.4s; }}
        
        .card:hover {{ 
            transform: translateY(-10px) scale(1.01);
            box-shadow: 0 15px 30px rgba(0,0,0,0.15);
            background: {hover_bg};
            border-color: {primary_color}4D;
        }}
        
        .card h3 {{
            color: {primary_color};
            font-size: 1.6em;
            margin-bottom: 1em;
            border-bottom: 2px solid {primary_color}33;
            padding-bottom: 0.5em;
            font-weight: 600;
        }}
        
        .card p, .card ul, .card ol {{
            color: {text_color};
            font-size: 1.1em;
            line-height: 1.7;
        }}
        
        /* Custom navigation */
        .custom-nav {{
            display: flex;
            background: {card_bg};
            padding: 10px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            overflow: hidden;
            border: 1px solid {card_border};
        }}
        
        .nav-item {{
            flex: 1;
            text-align: center;
            padding: 12px 20px;
            cursor: pointer;
            border-radius: 10px;
            margin: 0 5px;
            transition: all 0.3s ease;
            color: {text_color};
            font-weight: 500;
            position: relative;
            overflow: hidden;
        }}
        
        .nav-item:hover {{
            background: {primary_color}1A;
        }}
        
        .nav-item.active {{
            background: linear-gradient(135deg, {primary_color}, {secondary_color});
            color: white;
            box-shadow: 0 4px 15px {primary_color}4D;
        }}
        
        .nav-item.active::after {{
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3px;
            background: white;
        }}
        
        /* Notification panel */
        .notification-panel {{
            position: fixed;
            top: 70px;
            right: 20px;
            width: 300px;
            max-height: 400px;
            overflow-y: auto;
            background: {card_bg};
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
            z-index: 1000;
            border: 1px solid {card_border};
            padding: 15px;
            animation: slideInRight 0.3s ease-out;
        }}
        
        .notification-item {{
            padding: 10px;
            border-bottom: 1px solid {card_border};
            margin-bottom: 10px;
        }}
        
        .notification-item:last-child {{
            border-bottom: none;
            margin-bottom: 0;
        }}
        
        .notification-title {{
            font-weight: 600;
            color: {primary_color};
            margin-bottom: 5px;
        }}
        
        .notification-time {{
            font-size: 0.8em;
            color: {text_color}99;
        }}
        
        /* Theme toggle */
        .theme-toggle {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: {card_bg};
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            cursor: pointer;
            z-index: 1000;
            border: 1px solid {card_border};
            transition: all 0.3s ease;
        }}
        
        .theme-toggle:hover {{
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }}
        
        /* Dashboard widgets */
        .stat-card {{
            background: {card_bg};
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border: 1px solid {card_border};
            height: 100%;
            transition: all 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: 700;
            color: {primary_color};
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: {text_color};
            font-size: 1.1em;
            font-weight: 500;
        }}
        
        .stat-change {{
            display: flex;
            align-items: center;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        
        .stat-change.positive {{
            color: #10b981;
        }}
        
        .stat-change.negative {{
            color: #ef4444;
        }}
        
        /* Animation keyframes */
        @keyframes fadeInDown {{
            from {{
                opacity: 0;
                transform: translate3d(0, -50px, 0);
            }}
            to {{
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }}
        }}
        
        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translate3d(0, 50px, 0);
            }}
            to {{
                opacity: 1;
                transform: translate3d(0, 0, 0);
            }}
        }}
        
        @keyframes slideInRight {{
            from {{
                transform: translate3d(100%, 0, 0);
                opacity: 0;
            }}
            to {{
                transform: translate3d(0, 0, 0);
                opacity: 1;
            }}
        }}
        
        @keyframes pulse {{
            0% {{
                transform: scale(1);
            }}
            50% {{
                transform: scale(1.05);
            }}
            100% {{
                transform: scale(1);
            }}
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: {bg_color};
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: {primary_color}80;
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: {primary_color};
        }}
        
        /* Floating action button */
        .floating-button {{
            position: fixed;
            bottom: 80px;
            right: 20px;
            background: linear-gradient(135deg, {primary_color}, {secondary_color});
            color: white;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px {primary_color}4D;
            cursor: pointer;
            z-index: 1000;
            transition: all 0.3s ease;
        }}
        
        .floating-button:hover {{
            transform: scale(1.1) rotate(90deg);
            box-shadow: 0 6px 20px {primary_color}80;
        }}
        
        /* Custom tooltip */
        .tooltip {{
            position: relative;
            display: inline-block;
        }}
        
        .tooltip .tooltiptext {{
            visibility: hidden;
            width: 200px;
            background-color: {primary_color};
            color: white;
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
        }}
        
        .tooltip:hover .tooltiptext {{
            visibility: visible;
            opacity: 1;
        }}
        
        /* Custom data table */
        .custom-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        
        .custom-table th {{
            background: linear-gradient(135deg, {primary_color}, {secondary_color});
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        .custom-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid {card_border};
            color: {text_color};
        }}
        
        .custom-table tr:last-child td {{
            border-bottom: none;
        }}
        
        .custom-table tr:nth-child(even) {{
            background-color: {primary_color}0A;
        }}
        
        .custom-table tr:hover {{
            background-color: {primary_color}1A;
        }}
        
        /* Animated progress bar */
        .progress-container {{
            width: 100%;
            height: 10px;
            background-color: {card_border};
            border-radius: 5px;
            margin: 10px 0;
            overflow: hidden;
        }}
        
        .progress-bar {{
            height: 100%;
            background: linear-gradient(90deg, {primary_color}, {secondary_color});
            border-radius: 5px;
            transition: width 0.5s ease;
            animation: progress-animation 2s infinite;
        }}
        
        @keyframes progress-animation {{
            0% {{
                background-position: 0% 50%;
            }}
            50% {{
                background-position: 100% 50%;
            }}
            100% {{
                background-position: 0% 50%;
            }}
        }}
        
        /* Custom button */
        .custom-button {{
            background: linear-gradient(135deg, {primary_color}, {secondary_color});
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 10px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: inline-block;
            text-align: center;
            box-shadow: 0 4px 15px {primary_color}4D;
        }}
        
        .custom-button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 25px {primary_color}80;
        }}
        
        /* Custom input */
        .custom-input {{
            width: 100%;
            padding: 12px 15px;
            border-radius: 10px;
            border: 1px solid {card_border};
            background-color: {bg_color};
            color: {text_color};
            transition: all 0.3s ease;
        }}
        
        .custom-input:focus {{
            border-color: {primary_color};
            box-shadow: 0 0 0 3px {primary_color}33;
            outline: none;
        }}
        
        /* Custom select */
        .custom-select {{
            width: 100%;
            padding: 12px 15px;
            border-radius: 10px;
            border: 1px solid {card_border};
            background-color: {bg_color};
            color: {text_color};
            transition: all 0.3s ease;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='{primary_color}' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 15px center;
            background-size: 15px;
        }}
        
        .custom-select:focus {{
            border-color: {primary_color};
            box-shadow: 0 0 0 3px {primary_color}33;
            outline: none;
        }}
        
        /* Badge */
        .badge {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            margin-left: 10px;
        }}
        
        .badge-primary {{
            background-color: {primary_color};
            color: white;
        }}
        
        .badge-secondary {{
            background-color: {secondary_color};
            color: white;
        }}
        
        .badge-success {{
            background-color: #10b981;
            color: white;
        }}
        
        .badge-danger {{
            background-color: #ef4444;
            color: white;
        }}
        
        .badge-warning {{
            background-color: #f59e0b;
            color: white;
        }}
        
        .badge-info {{
            background-color: #3b82f6;
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

# Function to add notification
def add_notification(title, message):
    st.session_state.notifications.append({
        "title": title,
        "message": message,
        "time": datetime.now().strftime("%H:%M:%S")
    })

# Function to render notifications panel
def render_notifications():
    if st.session_state.notifications:
        notifications_html = "<div class='notification-panel'>"
        notifications_html += "<h3 style='margin-top:0;'>Notifications</h3>"
        
        for notification in st.session_state.notifications:
            notifications_html += f"""
            <div class='notification-item'>
                <div class='notification-title'>{notification['title']}</div>
                <div>{notification['message']}</div>
                <div class='notification-time'>{notification['time']}</div>
            </div>
            """
        
        notifications_html += "</div>"
        st.markdown(notifications_html, unsafe_allow_html=True)

# Function to render theme toggle
def render_theme_toggle():
    icon = "🌙" if not st.session_state.dark_mode else "☀️"
    toggle_html = f"""
    <div class='theme-toggle' onclick='toggleTheme()'>
        {icon}
    </div>
    <script>
    function toggleTheme() {{
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = '';

        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'theme_toggle';
        input.value = 'toggle';
        
        form.appendChild(input);
        document.body.appendChild(form);
        form.submit();
    }}
    </script>
    """
    st.markdown(toggle_html, unsafe_allow_html=True)

# Function to render floating action button
def render_floating_button():
    button_html = """
    <div class='floating-button' onclick='showHelp()'>
        ?
    </div>
    <script>
    function showHelp() {
        alert('Network Traffic Dashboard Help\\n\\nThis dashboard provides real-time insights into network traffic patterns. Use the navigation menu to explore different sections of the application.');
    }
    </script>
    """
    st.markdown(button_html, unsafe_allow_html=True)

# Function to render custom navigation
def render_navigation():
    tabs = {
        "dashboard": "📊 Dashboard",
        "analytics": "📈 Analytics",
        "settings": "⚙️ Settings",
        "help": "❓ Help"
    }
    
    nav_html = "<div class='custom-nav'>"
    
    for tab_id, tab_name in tabs.items():
        active_class = "active" if st.session_state.active_tab == tab_id else ""
        nav_html += f"""
        <div class='nav-item {active_class}' onclick='navigateTo("{tab_id}")'>
            {tab_name}
        </div>
        """
    
    nav_html += "</div>"
    
    nav_html += """
    <script>
    function navigateTo(tabId) {
        const form = document.createElement('form');
        form.method = 'POST';
        form.action = '';

        const input = document.createElement('input');
        input.type = 'hidden';
        input.name = 'active_tab';
        input.value = tabId;
        
        form.appendChild(input);
        document.body.appendChild(form);
        form.submit();
    }
    </script>
    """
    
    st.markdown(nav_html, unsafe_allow_html=True)

# Function to render dashboard
def render_dashboard():
    st.markdown('<h1 class="main-header">🌐 Network Traffic Dashboard</h1>', unsafe_allow_html=True)
    
    # Dashboard stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Total Traffic</div>
            <div class="stat-value">1.28 TB</div>
            <div class="stat-change positive">
                ↑ 12.5% from last week
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Active Connections</div>
            <div class="stat-value">2,845</div>
            <div class="stat-change positive">
                ↑ 5.2% from yesterday
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Avg. Response Time</div>
            <div class="stat-value">42 ms</div>
            <div class="stat-change negative">
                ↓ 3.1% from yesterday
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Security Alerts</div>
            <div class="stat-value">17</div>
            <div class="stat-change negative">
                ↑ 8.4% from yesterday
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Traffic chart
    st.markdown("<div class='card'><h3>Network Traffic Overview</h3>", unsafe_allow_html=True)
    
    # Generate sample data for the chart
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    incoming = np.random.normal(500, 100, 30).cumsum()
    outgoing = np.random.normal(300, 80, 30).cumsum()
    
    df = pd.DataFrame({
        'Date': dates,
        'Incoming': incoming,
        'Outgoing': outgoing
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Incoming'],
        mode='lines',
        name='Incoming Traffic',
        line=dict(width=3, color='#3a7bd5'),
        fill='tozeroy',
        fillcolor='rgba(58, 123, 213, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Outgoing'],
        mode='lines',
        name='Outgoing Traffic',
        line=dict(width=3, color='#00d2ff'),
        fill='tozeroy',
        fillcolor='rgba(0, 210, 255, 0.2)'
    ))
    
    fig.update_layout(
        title='',
        xaxis_title='Date',
        yaxis_title='Traffic Volume (GB)',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Traffic distribution and alerts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'><h3>Traffic Distribution</h3>", unsafe_allow_html=True)
        
        # Sample data for traffic distribution
        traffic_types = ['HTTP', 'HTTPS', 'FTP', 'SSH', 'DNS', 'SMTP', 'Other']
        traffic_values = [45, 30, 10, 5, 5, 3, 2]
        
        fig = px.pie(
            names=traffic_types,
            values=traffic_values,
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'><h3>Recent Security Alerts</h3>", unsafe_allow_html=True)
        
        alerts = [
            {"time": "10:45 AM", "type": "Suspicious Login", "severity": "High", "status": "Open"},
            {"time": "09:32 AM", "type": "Port Scan", "severity": "Medium", "status": "Investigating"},
            {"time": "08:17 AM", "type": "DDoS Attempt", "severity": "High", "status": "Resolved"},
            {"time": "Yesterday", "type": "Malware Detected", "severity": "Critical", "status": "Resolved"},
        ]
        
        alerts_html = "<table class='custom-table'>"
        alerts_html += "<tr><th>Time</th><th>Type</th><th>Severity</th><th>Status</th></tr>"
        
        for alert in alerts:
            severity_class = "danger" if alert["severity"] in ["High", "Critical"] else "warning" if alert["severity"] == "Medium" else "info"
            status_class = "success" if alert["status"] == "Resolved" else "warning" if alert["status"] == "Investigating" else "danger"
            
            alerts_html += f"""
            <tr>
                <td>{alert["time"]}</td>
                <td>{alert["type"]}</td>
                <td><span class="badge badge-{severity_class}">{alert["severity"]}</span></td>
                <td><span class="badge badge-{status_class}">{alert["status"]}</span></td>
            </tr>
            """
        
        alerts_html += "</table>"
        st.markdown(alerts_html, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <a href="#" class="custom-button">View All Alerts</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Network health
    st.markdown("<div class='card'><h3>Network Health</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <h4 style="color: #3a7bd5;">CPU Usage</h4>
        <div class="progress-container">
            <div class="progress-bar" style="width: 65%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>65%</span>
            <span>Normal</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <h4 style="color: #3a7bd5;">Memory Usage</h4>
        <div class="progress-container">
            <div class="progress-bar" style="width: 78%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>78%</span>
            <span>Warning</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <h4 style="color: #3a7bd5;">Disk Usage</h4>
        <div class="progress-container">
            <div class="progress-bar" style="width: 42%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>42%</span>
            <span>Normal</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Function to render analytics
def render_analytics():
    st.markdown('<h1 class="main-header">📈 Network Analytics</h1>', unsafe_allow_html=True)
    
    # Analytics filters
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <label for="time-range" style="font-weight: 500; color: #3a7bd5;">Time Range</label>
        <select id="time-range" class="custom-select">
            <option value="today">Today</option>
            <option value="yesterday">Yesterday</option>
            <option value="last7" selected>Last 7 Days</option>
            <option value="last30">Last 30 Days</option>
            <option value="custom">Custom Range</option>
        </select>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <label for="traffic-type" style="font-weight: 500; color: #3a7bd5;">Traffic Type</label>
        <select id="traffic-type" class="custom-select">
            <option value="all" selected>All Traffic</option>
            <option value="http">HTTP/HTTPS</option>
            <option value="ftp">FTP</option>
            <option value="ssh">SSH</option>
            <option value="dns">DNS</option>
        </select>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <label for="group-by" style="font-weight: 500; color: #3a7bd5;">Group By</label>
        <select id="group-by" class="custom-select">
            <option value="hour">Hour</option>
            <option value="day" selected>Day</option>
            <option value="week">Week</option>
            <option value="month">Month</option>
        </select>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <a href="#" class="custom-button">Apply Filters</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Traffic trends
    st.markdown("<div class='card'><h3>Traffic Trends</h3>", unsafe_allow_html=True)
    
    # Generate sample data for traffic trends
    dates = pd.date_range(start='2023-01-01', periods=7, freq='D')
    http = np.random.normal(500, 50, 7)
    https = np.random.normal(800, 80, 7)
    ftp = np.random.normal(200, 30, 7)
    ssh = np.random.normal(100, 20, 7)
    dns = np.random.normal(300, 40, 7)
    
    df = pd.DataFrame({
        'Date': dates,
        'HTTP': http,
        'HTTPS': https,
        'FTP': ftp,
        'SSH': ssh,
        'DNS': dns
    })
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['HTTP'],
        name='HTTP',
        marker_color='#3a7bd5'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['HTTPS'],
        name='HTTPS',
        marker_color='#00d2ff'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['FTP'],
        name='FTP',
        marker_color='#10b981'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['SSH'],
        name='SSH',
        marker_color='#f59e0b'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Date'],
        y=df['DNS'],
        name='DNS',
        marker_color='#ef4444'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='',
        xaxis_title='Date',
        yaxis_title='Traffic Volume (GB)',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Geographic distribution
    st.markdown("<div class='card'><h3>Geographic Distribution</h3>", unsafe_allow_html=True)
    
    # Sample data for geographic distribution
    countries = ['United States', 'China', 'India', 'Germany', 'United Kingdom', 'Brazil', 'Japan', 'Canada', 'France', 'Australia']
    traffic = [35, 20, 15, 8, 7, 5, 4, 3, 2, 1]
    
    df = pd.DataFrame({
        'Country': countries,
        'Traffic': traffic
    })
    
    fig = px.choropleth(
        df,
        locations='Country',
        locationmode='country names',
        color='Traffic',
        hover_name='Country',
        color_continuous_scale='Blues',
        projection='natural earth'
    )
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'><h3>Response Time</h3>", unsafe_allow_html=True)
        
        # Generate sample data for response time
        dates = pd.date_range(start='2023-01-01', periods=7, freq='D')
        response_time = np.random.normal(50, 10, 7)
        
        fig = px.line(
            x=dates,
            y=response_time,
            markers=True,
            line_shape='spline',
            labels={'x': 'Date', 'y': 'Response Time (ms)'},
            color_discrete_sequence=['#3a7bd5']
        )
        
        fig.update_layout(
            title='',
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'><h3>Error Rate</h3>", unsafe_allow_html=True)
        
        # Generate sample data for error rate
        dates = pd.date_range(start='2023-01-01', periods=7, freq='D')
        error_rate = np.random.normal(2, 0.5, 7)
        
        fig = px.line(
            x=dates,
            y=error_rate,
            markers=True,
            line_shape='spline',
            labels={'x': 'Date', 'y': 'Error Rate (%)'},
            color_discrete_sequence=['#ef4444']
        )
        
        fig.update_layout(
            title='',
            template='plotly_white',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Function to render settings
def render_settings():
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    
    st.markdown("<div class='card'><h3>Display Settings</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <label for="theme" style="font-weight: 500; color: #3a7bd5;">Theme</label>
        <select id="theme" class="custom-select">
            <option value="light" selected>Light</option>
            <option value="dark">Dark</option>
            <option value="auto">Auto (System)</option>
        </select>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <label for="refresh-rate" style="font-weight: 500; color: #3a7bd5;">Data Refresh Rate</label>
        <select id="refresh-rate" class="custom-select">
            <option value="30">30 seconds</option>
            <option value="60" selected>1 minute</option>
            <option value="300">5 minutes</option>
            <option value="600">10 minutes</option>
            <option value="1800">30 minutes</option>
        </select>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'><h3>Notification Settings</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <label style="display: flex; align-items: center; margin-bottom: 15px;">
            <input type="checkbox" checked style="margin-right: 10px; width: 18px; height: 18px;">
            <span style="font-weight: 500;">Enable Email Notifications</span>
        </label>
        
        <label style="display: flex; align-items: center; margin-bottom: 15px;">
            <input type="checkbox" checked style="margin-right: 10px; width: 18px; height: 18px;">
            <span style="font-weight: 500;">Enable In-App Notifications</span>
        </label>
        
        <label style="display: flex; align-items: center; margin-bottom: 15px;">
            <input type="checkbox" style="margin-right: 10px; width: 18px; height: 18px;">
            <span style="font-weight: 500;">Enable SMS Notifications</span>
        </label>
    </div>
    
    <label for="email" style="font-weight: 500; color: #3a7bd5;">Email Address</label>
    <input type="email" id="email" class="custom-input" value="admin@example.com" style="margin-bottom: 15px;">
    
    <label for="phone" style="font-weight: 500; color: #3a7bd5;">Phone Number</label>
    <input type="tel" id="phone" class="custom-input" placeholder="+1 (123) 456-7890" style="margin-bottom: 15px;">
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'><h3>Alert Thresholds</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <label for="cpu-threshold" style="font-weight: 500; color: #3a7bd5;">CPU Usage Threshold (%)</label>
        <input type="number" id="cpu-threshold" class="custom-input" value="80" min="0" max="100" style="margin-bottom: 15px;">
        
        <label for="memory-threshold" style="font-weight: 500; color: #3a7bd5;">Memory Usage Threshold (%)</label>
        <input type="number" id="memory-threshold" class="custom-input" value="85" min="0" max="100" style="margin-bottom: 15px;">
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <label for="disk-threshold" style="font-weight: 500; color: #3a7bd5;">Disk Usage Threshold (%)</label>
        <input type="number" id="disk-threshold" class="custom-input" value="90" min="0" max="100" style="margin-bottom: 15px;">
        
        <label for="response-threshold" style="font-weight: 500; color: #3a7bd5;">Response Time Threshold (ms)</label>
        <input type="number" id="response-threshold" class="custom-input" value="200" min="0" style="margin-bottom: 15px;">
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 20px;">
        <a href="#" class="custom-button">Save Settings</a>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Function to render help
def render_help():
    st.markdown('<h1 class="main-header">❓ Help & Support</h1>', unsafe_allow_html=True)
    
    # Load help animation
    lottie_help = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json")
    if lottie_help:
        st_lottie(lottie_help, height=250, key="help_animation")
    
    st.markdown("<div class='card'><h3>Quick Start Guide</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <p>Welcome to the Network Traffic Dashboard! Here's how to get started:</p>
    
    <ol>
        <li><strong>Dashboard:</strong> View real-time network traffic statistics and alerts</li>
        <li><strong>Analytics:</strong> Explore detailed traffic trends and performance metrics</li>
        <li><strong>Settings:</strong> Configure notification preferences and alert thresholds</li>
        <li><strong>Help:</strong> Access documentation and support resources</li>
    </ol>
    
    <p>Use the navigation menu at the top to switch between different sections of the application.</p>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='card'><h3>Frequently Asked Questions</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <p style="font-weight: 600; color: #3a7bd5;">How is network traffic data collected?</p>
        <p>Network traffic data is collected through a combination of packet sniffing, flow analysis, and system logs. The data is processed in real-time to provide up-to-date insights.</p>
    </div>
    
    <div style="margin-bottom: 20px;">
        <p style="font-weight: 600; color: #3a7bd5;">How often is the dashboard updated?</p>
        <p>By default, the dashboard refreshes every minute. You can adjust the refresh rate in the Settings section.</p>
    </div>
    
    <div style="margin-bottom: 20px;">
        <p style="font-weight: 600; color: #3a7bd5;">How do I export data for reporting?</p>
        <p>You can export data in CSV or PDF format using the download buttons available in the Analytics section.</p>
    </div>
    
    <div style="margin-bottom: 20px;">
        <p style="font-weight: 600; color: #3a7bd5;">What do the different alert severities mean?</p>
        <p>Alert severities are categorized as follows:</p>
        <ul>
            <li><strong>Critical:</strong> Requires immediate attention</li>
            <li><strong>High:</strong> Should be addressed within hours</li>
            <li><strong>Medium:</strong> Should be investigated within a day</li>
            <li><strong>Low:</strong> Informational, no immediate action required</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'><h3>Contact Support</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <p>Need help? Contact our support team:</p>
        
        <p><strong>Email:</strong> support@networktraffic.ai</p>
        <p><strong>Phone:</strong> +1 (123) 456-7890</p>
        <p><strong>Hours:</strong> Monday-Friday, 9am-5pm EST</p>
        
        <div style="margin-top: 20px;">
            <label for="subject" style="font-weight: 500; color: #3a7bd5;">Subject</label>
            <input type="text" id="subject" class="custom-input" placeholder="Enter subject" style="margin-bottom: 15px;">
            
            <label for="message" style="font-weight: 500; color: #3a7bd5;">Message</label>
            <textarea id="message" class="custom-input" placeholder="Enter your message" style="height: 150px; margin-bottom: 15px;"></textarea>
            
            <div style="text-align: center;">
                <a href="#" class="custom-button">Send Message</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'><h3>Documentation</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <p>Access comprehensive documentation for the Network Traffic Dashboard:</p>
        
        <div style="margin: 20px 0;">
            <a href="#" class="custom-button" style="display: block; margin-bottom: 15px;">User Guide</a>
            <a href="#" class="custom-button" style="display: block; margin-bottom: 15px;">API Documentation</a>
            <a href="#" class="custom-button" style="display: block; margin-bottom: 15px;">Release Notes</a>
            <a href="#" class="custom-button" style="display: block;">Troubleshooting Guide</a>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def main():
    # Load CSS
    load_css()
    
    # Check for form submissions
    if st.experimental_get_query_params():
        query_params = st.experimental_get_query_params()
        
        if 'theme_toggle' in query_params:
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.experimental_set_query_params()
            st.experimental_rerun()
        
        if 'active_tab' in query_params:
            st.session_state.active_tab = query_params['active_tab'][0]
            st.experimental_set_query_params()
            st.experimental_rerun()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 style="text-align: center; background: linear-gradient(90deg, #3a7bd5, #00d2ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; margin-bottom: 30px;">🌐 Network AI</h2>', unsafe_allow_html=True)
        
        # Load sidebar animation
        lottie_sidebar = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qp1q7mct.json")
        if lottie_sidebar:
            st_lottie(lottie_sidebar, height=200, key="sidebar_animation")
        
        # Add a notification for demo purposes
        if len(st.session_state.notifications) == 0:
            add_notification("Welcome", "Welcome to the Network Traffic Dashboard!")
            add_notification("New Update", "Dashboard version 2.0 is now available.")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 15px; background: rgba(58, 123, 213, 0.1); border-radius: 10px; margin-top: 20px;">
            <p style="color: #3a7bd5; font-weight: 500;">Network Traffic Dashboard v2.0</p>
            <p style="font-size: 0.8em; color: #34495e;">Last updated: April 4, 2025</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Render custom navigation
    render_navigation()
    
    # Render content based on active tab
    if st.session_state.active_tab == "dashboard":
        render_dashboard()
    elif st.session_state.active_tab == "analytics":
        render_analytics()
    elif st.session_state.active_tab == "settings":
        render_settings()
    elif st.session_state.active_tab == "help":
        render_help()
    
    # Render notifications panel
    render_notifications()
    
    # Render theme toggle
    render_theme_toggle()
    
    # Render floating action button
    render_floating_button()

if __name__ == "__main__":
    main()
