import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Al-Ittihad Tripoli - Performance Analysis", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .status-green {
            background-color: #27ae60;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            text-align: center;
            font-weight: bold;
            display: inline-block;
        }
        .status-yellow {
            background-color: #f39c12;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            text-align: center;
            font-weight: bold;
            display: inline-block;
        }
        .status-red {
            background-color: #e74c3c;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            text-align: center;
            font-weight: bold;
            display: inline-block;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header"><h1 style="color: white;">AL-ITTIHAD TRIPOLI</h1><h3 style="color: white;">Player Performance & Injury Risk Analysis System</h3><p style="color: white;">For Coaching Staff | Data-Driven Insights</p></div>', unsafe_allow_html=True)

def load_training_data(uploaded_file):
    """Load training data from CSV file with proper structure"""
    try:
        # Read the CSV file, handling quotes
        # First, read all lines to find where data starts
        content = uploaded_file.getvalue().decode('utf-8')
        lines = content.split('\n')
        
        # Find the line with column headers (contains "Player Name")
        header_line_idx = None
        for i, line in enumerate(lines):
            if 'Player Name' in line and 'Period Name' in line:
                header_line_idx = i
                break
        
        if header_line_idx is None:
            st.error("Could not find data headers in file")
            return None, None
        
        # Read from the header line onwards
        data_content = '\n'.join(lines[header_line_idx:])
        
        # Read CSV with proper handling of quotes
        df = pd.read_csv(io.StringIO(data_content), encoding='utf-8')
        
        # Clean column names (remove quotes and extra spaces)
        df.columns = df.columns.str.strip().str.replace('"', '')
        
        # Filter to only include rows with valid player names and Session period
        df = df[df['Player Name'].notna()]
        df = df[df['Player Name'].astype(str).str.strip() != '']
        df = df[df['Player Name'] != 'Player Name']
        
        # Keep only rows where Period Name is 'Session' (not AutoCreatedPeriod)
        if 'Period Name' in df.columns:
            df = df[df['Period Name'] == 'Session']
        
        # Convert numeric columns
        numeric_columns = [
            'Average Distance (Session)', 'Average Player Load (Session)',
            'Maximum Velocity', 'Meterage Per Minute', 'Player Load Per Minute',
            'Accel + Decel Efforts', 'Accel + Decel Efforts Per Minute',
            'High Metabolic Load Distance', 'HS Distance', 'HS Dist Per Min',
            'Sprint Dist Per Min', 'Sprint Efforts'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract session info from filename
        session_name = uploaded_file.name.replace('.csv', '')
        
        return df, session_name
    except Exception as e:
        st.error(f"Error loading file {uploaded_file.name}: {str(e)}")
        return None, None

def get_wellness_status(player_name, avg_load):
    """Determine wellness status based on player load and PDF data"""
    # Wellness data from the PDF provided
    wellness_dict = {
        "Hedi AL Snoussi": "Yellow",
        "Anas Al Shebli": "Yellow", 
        "Mohamed Al Zantani": "Yellow",
        "Hussain Badi": "Green",
        "Taher Ben Amer": "Yellow",
        "Abdulmuyasser Bouchiba": "Yellow",
        "Mahmoud Cheloui": "Green",
        "LORCH Chris": "Yellow",
        "Mohamed Chtioui": "Green",
        "Talal Farhat": "Green",
        "Youssef Kara": "Green",
        "Mootassem Sabbou": "Yellow",
        "Fadhel Salama": "Green",
        "Sobhi EL Mabouk": "Green",
        "Aziz Ki": "Green"
    }
    
    if player_name in wellness_dict:
        return wellness_dict[player_name]
    else:
        # Default logic based on load
        if pd.notna(avg_load):
            if avg_load > 450:
                return "Red"
            elif avg_load > 350:
                return "Yellow"
            else:
                return "Green"
        return "Green"

# Sidebar for file upload and session selection
with st.sidebar:
    st.header("Data Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Training Session CSV Files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload CSV files from Catapult system"
    )
    
    st.markdown("---")
    
    # Session selection
    if uploaded_files:
        st.header("Session Selection")
        session_names = []
        for f in uploaded_files:
            session_name = f.name.replace('.csv', '').replace('ctr-report-activity-', '')
            session_names.append(session_name)
        
        selected_sessions = st.multiselect(
            "Select sessions to analyze",
            session_names,
            default=session_names[:min(2, len(session_names))]
        )
    
    st.markdown("---")
    st.header("Analysis Settings")
    risk_threshold = st.slider(
        "Injury Risk Threshold",
        0.0, 1.0, 0.5,
        help="Higher threshold = stricter injury prediction"
    )
    
    st.markdown("---")
    st.header("About")
    st.info("""
    **Al-Ittihad Tripoli Performance System**
    
    This system analyzes player performance metrics:
    - Distance covered
    - Player load
    - Maximum velocity
    - Accelerations/Decelerations
    - ML-based injury risk prediction
    """)

# Import io for string buffer
import io

# Main content
if uploaded_files and 'selected_sessions' in locals() and selected_sessions:
    # Load all training data
    all_data = {}
    session_data = {}
    
    for file in uploaded_files:
        session_name = file.name.replace('.csv', '').replace('ctr-report-activity-', '')
        if session_name in selected_sessions:
            df, _ = load_training_data(file)
            if df is not None and len(df) > 0:
                all_data[session_name] = df
                session_data[session_name] = df
                st.success(f"Loaded {len(df)} players from {session_name}")
    
    if all_data:
        # Combine data from all sessions
        combined_df = pd.concat(all_data.values(), ignore_index=True)
        players = combined_df['Player Name'].unique().tolist()
        
        # Calculate player averages across sessions
        player_avg = combined_df.groupby('Player Name').agg({
            'Average Distance (Session)': 'mean',
            'Average Player Load (Session)': 'mean',
            'Maximum Velocity': 'max',
            'Meterage Per Minute': 'mean',
            'Accel + Decel Efforts': 'mean',
            'High Metabolic Load Distance': 'mean'
        }).reset_index()
        
        # Add wellness status
        player_avg['Wellness Status'] = player_avg.apply(
            lambda x: get_wellness_status(x['Player Name'], x['Average Player Load (Session)']), 
            axis=1
        )
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Player Performance",
            "📈 Metric Comparison", 
            "⚠️ Injury Risk Prediction",
            "📋 Player Status Report"
        ])
        
        # Tab 1: Player Performance
        with tab1:
            st.header("Individual Player Performance Analysis")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                selected_player = st.selectbox("Select Player", players)
                selected_metric = st.selectbox(
                    "Select Metric",
                    ['Average Distance (Session)', 'Average Player Load (Session)',
                     'Maximum Velocity', 'Meterage Per Minute', 'Accel + Decel Efforts']
                )
            
            if selected_player:
                with col2:
                    player_info = player_avg[player_avg['Player Name'] == selected_player].iloc[0]
                    status = player_info['Wellness Status']
                    status_color = f"status-{status.lower()}"
                    st.markdown(f'<div class="{status_color}">Wellness Status: {status}</div>', unsafe_allow_html=True)
                
                # Performance across sessions
                player_session_data = []
                for session_name, df in session_data.items():
                    player_row = df[df['Player Name'] == selected_player]
                    if len(player_row) > 0:
                        distance_val = player_row['Average Distance (Session)'].values[0] if pd.notna(player_row['Average Distance (Session)'].values[0]) else 0
                        load_val = player_row['Average Player Load (Session)'].values[0] if pd.notna(player_row['Average Player Load (Session)'].values[0]) else 0
                        vel_val = player_row['Maximum Velocity'].values[0] if pd.notna(player_row['Maximum Velocity'].values[0]) else 0
                        meter_val = player_row['Meterage Per Minute'].values[0] if pd.notna(player_row['Meterage Per Minute'].values[0]) else 0
                        accel_val = player_row['Accel + Decel Efforts'].values[0] if pd.notna(player_row['Accel + Decel Efforts'].values[0]) else 0
                        
                        player_session_data.append({
                            'Session': session_name,
                            'Distance': distance_val,
                            'Player Load': load_val,
                            'Max Velocity': vel_val,
                            'Meterage/Min': meter_val,
                            'Accel/Decel': accel_val
                        })
                
                if player_session_data:
                    session_df = pd.DataFrame(player_session_data)
                    
                    # Create performance chart
                    fig = go.Figure()
                    
                    if selected_metric == 'Average Distance (Session)':
                        values = session_df['Distance']
                    elif selected_metric == 'Average Player Load (Session)':
                        values = session_df['Player Load']
                    elif selected_metric == 'Maximum Velocity':
                        values = session_df['Max Velocity']
                    elif selected_metric == 'Meterage Per Minute':
                        values = session_df['Meterage/Min']
                    elif selected_metric == 'Accel + Decel Efforts':
                        values = session_df['Accel/Decel']
                    else:
                        values = session_df['Distance']
                    
                    fig.add_trace(go.Bar(
                        name=selected_metric,
                        x=session_df['Session'],
                        y=values,
                        text=values.round(1),
                        textposition='auto',
                        marker_color='#3498db'
                    ))
                    fig.update_layout(
                        title=f"{selected_player} - {selected_metric} Across Sessions",
                        xaxis_title="Training Session",
                        yaxis_title=selected_metric,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("Detailed Performance Metrics")
                detail_metrics = ['Average Distance (Session)', 'Average Player Load (Session)',
                                'Maximum Velocity', 'Meterage Per Minute', 'Player Load Per Minute',
                                'Accel + Decel Efforts', 'High Metabolic Load Distance']
                
                available_metrics = [m for m in detail_metrics if m in combined_df.columns]
                player_detail = combined_df[combined_df['Player Name'] == selected_player][available_metrics]
                
                if len(player_detail) > 0:
                    player_detail.index = [f"Session {i+1}" for i in range(len(player_detail))]
                    st.dataframe(player_detail.round(1), use_container_width=True)
        
        # Tab 2: Metric Comparison
        with tab2:
            st.header("Player Performance Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                compare_metric = st.selectbox(
                    "Select Metric to Compare",
                    ['Average Distance (Session)', 'Average Player Load (Session)',
                     'Maximum Velocity', 'Meterage Per Minute', 'Accel + Decel Efforts'],
                    key="compare_metric"
                )
            
            with col2:
                sort_order = st.selectbox("Sort Order", ["Descending", "Ascending"])
            
            # Create comparison chart
            comparison_data = player_avg[['Player Name', compare_metric, 'Wellness Status']].dropna()
            comparison_data = comparison_data.sort_values(
                compare_metric, 
                ascending=(sort_order == "Ascending")
            )
            
            # Color mapping based on wellness
            color_map = {'Green': '#27ae60', 'Yellow': '#f39c12', 'Red': '#e74c3c'}
            
            fig = px.bar(
                comparison_data,
                x='Player Name',
                y=compare_metric,
                color='Wellness Status',
                color_discrete_map=color_map,
                title=f"Player Comparison - {compare_metric}",
                text=comparison_data[compare_metric].round(1)
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top performers table
            st.subheader("Top Performers")
            col1, col2 = st.columns(2)
            
            with col1:
                top_distance = player_avg.nlargest(5, 'Average Distance (Session)')[['Player Name', 'Average Distance (Session)', 'Wellness Status']]
                st.markdown("**Top 5 by Distance**")
                st.dataframe(top_distance.round(1), use_container_width=True)
            
            with col2:
                top_load = player_avg.nlargest(5, 'Average Player Load (Session)')[['Player Name', 'Average Player Load (Session)', 'Wellness Status']]
                st.markdown("**Top 5 by Player Load**")
                st.dataframe(top_load.round(1), use_container_width=True)
        
        # Tab 3: Injury Risk Prediction
        with tab3:
            st.header("Machine Learning Injury Risk Prediction")
            st.markdown("Predicts injury probability based on training load and performance metrics")
            
            # Prepare features for ML
            feature_cols = ['Average Distance (Session)', 'Average Player Load (Session)',
                          'Maximum Velocity', 'Meterage Per Minute', 'Accel + Decel Efforts']
            
            available_features = [col for col in feature_cols if col in combined_df.columns]
            
            if len(available_features) >= 3:
                # Aggregate by player
                feature_data = combined_df.groupby('Player Name')[available_features].mean().reset_index()
                feature_data = feature_data.dropna()
                
                # Create risk labels based on load and wellness
                feature_data['risk_score'] = 0
                for idx, row in feature_data.iterrows():
                    load = row['Average Player Load (Session)']
                    if pd.notna(load):
                        wellness = get_wellness_status(row['Player Name'], load)
                        if wellness == 'Red' or load > 480:
                            feature_data.loc[idx, 'risk_score'] = 1
                        elif wellness == 'Yellow' or load > 400:
                            feature_data.loc[idx, 'risk_score'] = 0.5
                        else:
                            feature_data.loc[idx, 'risk_score'] = 0
                
                # Prepare data for ML
                X = feature_data[available_features].fillna(0)
                y = (feature_data['risk_score'] >= risk_threshold).astype(int)
                
                # Train model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_scaled, y)
                
                # Predict
                probabilities = model.predict_proba(X_scaled)[:, 1]
                
                # Create risk dataframe
                risk_data = feature_data[['Player Name']].copy()
                risk_data['Injury Risk Probability'] = probabilities
                risk_data['Risk Level'] = risk_data['Injury Risk Probability'].apply(
                    lambda x: 'High Risk' if x > 0.7 else ('Moderate Risk' if x > 0.3 else 'Low Risk')
                )
                
                # Add wellness status
                risk_data['Wellness'] = risk_data['Player Name'].apply(
                    lambda x: get_wellness_status(x, feature_data[feature_data['Player Name'] == x]['Average Player Load (Session)'].values[0] if len(feature_data[feature_data['Player Name'] == x]) > 0 else 0)
                )
                
                risk_data = risk_data.sort_values('Injury Risk Probability', ascending=False)
                
                # Display risk table
                st.subheader("Injury Risk Assessment")
                
                def color_risk_cell(val):
                    if 'High' in str(val):
                        return 'background-color: #ffcccc'
                    elif 'Moderate' in str(val):
                        return 'background-color: #fff3cd'
                    elif 'Low' in str(val):
                        return 'background-color: #d4edda'
                    return ''
                
                styled_risk = risk_data.style.applymap(color_risk_cell, subset=['Risk Level'])
                st.dataframe(styled_risk, use_container_width=True)
                
                # Risk distribution chart
                col1, col2 = st.columns(2)
                
                with col1:
                    risk_counts = risk_data['Risk Level'].value_counts()
                    fig = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Team Risk Distribution",
                        color_discrete_sequence=['#27ae60', '#f39c12', '#e74c3c']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        risk_data,
                        x='Player Name',
                        y='Injury Risk Probability',
                        color='Risk Level',
                        title="Individual Risk Assessment",
                        color_discrete_map={'Low Risk': '#27ae60', 'Moderate Risk': '#f39c12', 'High Risk': '#e74c3c'}
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance
                st.subheader("Risk Factors Analysis")
                importance_df = pd.DataFrame({
                    'Feature': available_features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Factors Contributing to Injury Risk",
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("Coaching Recommendations")
                for _, row in risk_data.iterrows():
                    if row['Risk Level'] == 'High Risk':
                        st.warning(f"**{row['Player Name']}**: HIGH INJURY RISK - Immediate load reduction recommended, close monitoring required")
                    elif row['Risk Level'] == 'Moderate Risk':
                        st.info(f"**{row['Player Name']}: MODERATE RISK - Monitor carefully, consider load adjustment")
                    else:
                        st.success(f"**{row['Player Name']}**: LOW RISK - Continue current training program")
        
        # Tab 4: Player Status Report
        with tab4:
            st.header("Player Status Report")
            
            # Create comprehensive status table
            status_data = []
            for player in players:
                player_info = player_avg[player_avg['Player Name'] == player]
                if len(player_info) > 0:
                    avg_load = player_info['Average Player Load (Session)'].values[0]
                    avg_distance = player_info['Average Distance (Session)'].values[0]
                    max_vel = player_info['Maximum Velocity'].values[0]
                    wellness = player_info['Wellness Status'].values[0]
                    
                    # Determine performance level
                    if pd.notna(avg_load):
                        if avg_load > 450:
                            performance = "High Intensity"
                        elif avg_load > 350:
                            performance = "Moderate Intensity"
                        else:
                            performance = "Low Intensity"
                    else:
                        performance = "N/A"
                    
                    # Overall status
                    if wellness == "Red" or (pd.notna(avg_load) and avg_load > 480):
                        overall = "Red"
                    elif wellness == "Yellow" or (pd.notna(avg_load) and avg_load > 400):
                        overall = "Yellow"
                    else:
                        overall = "Green"
                    
                    status_data.append({
                        "Player Name": player,
                        "Wellness Status": wellness,
                        "Performance Level": performance,
                        "Avg Load": f"{avg_load:.0f}" if pd.notna(avg_load) else "N/A",
                        "Avg Distance": f"{avg_distance:.0f}" if pd.notna(avg_distance) else "N/A",
                        "Max Velocity": f"{max_vel:.1f}" if pd.notna(max_vel) else "N/A",
                        "Overall Status": overall
                    })
            
            status_df = pd.DataFrame(status_data)
            
            # Display status table with styling
            def color_overall(val):
                if val == 'Green':
                    return 'background-color: #27ae60; color: white'
                elif val == 'Yellow':
                    return 'background-color: #f39c12; color: white'
                elif val == 'Red':
                    return 'background-color: #e74c3c; color: white'
                return ''
            
            styled_df = status_df.style.applymap(color_overall, subset=['Overall Status'])
            st.dataframe(styled_df, use_container_width=True)
            
            # Team summary
            st.subheader("Team Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                green_count = len(status_df[status_df['Overall Status'] == 'Green'])
                st.metric("Ready to Play", green_count, f"{green_count/len(players)*100:.0f}%")
            
            with col2:
                yellow_count = len(status_df[status_df['Overall Status'] == 'Yellow'])
                st.metric("Monitor Closely", yellow_count, f"{yellow_count/len(players)*100:.0f}%")
            
            with col3:
                red_count = len(status_df[status_df['Overall Status'] == 'Red'])
                st.metric("High Risk", red_count, f"{red_count/len(players)*100:.0f}%")
            
            with col4:
                load_values = status_df[status_df['Avg Load'] != 'N/A']['Avg Load'].astype(float)
                if len(load_values) > 0:
                    avg_team_load = load_values.mean()
                    st.metric("Avg Team Load", f"{avg_team_load:.0f}")
                else:
                    st.metric("Avg Team Load", "N/A")
            
            # Export options
            st.subheader("Export Report")
            col1, col2 = st.columns(2)
            
            with col1:
                csv = status_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="al_ittihad_status_report.csv" style="text-decoration: none;"><button style="background-color: #3498db; color: white; padding: 0.5rem 1rem; border: none; border-radius: 5px; cursor: pointer;">📥 Download CSV Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                # Create text report
                report_text = f"""AL-ITTIHAD TRIPOLI - PLAYER STATUS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Players: {len(players)}

SUMMARY STATISTICS:
- Green Status (Ready): {green_count} players ({green_count/len(players)*100:.0f}%)
- Yellow Status (Monitor): {yellow_count} players ({yellow_count/len(players)*100:.0f}%)
- Red Status (High Risk): {red_count} players ({red_count/len(players)*100:.0f}%)
- Average Team Load: {avg_team_load:.0f}

PLAYER DETAILS:
{status_df.to_string(index=False)}

RECOMMENDATIONS:
"""
                for _, row in status_df.iterrows():
                    if row['Overall Status'] == 'Red':
                        report_text += f"\n- {row['Player Name']}: HIGH RISK - Immediate load reduction required"
                    elif row['Overall Status'] == 'Yellow':
                        report_text += f"\n- {row['Player Name']}: MODERATE RISK - Monitor and consider load adjustment"
                
                b64_text = base64.b64encode(report_text.encode()).decode()
                href_text = f'<a href="data:text/plain;base64,{b64_text}" download="al_ittihad_status_report.txt" style="text-decoration: none;"><button style="background-color: #2ecc71; color: white; padding: 0.5rem 1rem; border: none; border-radius: 5px; cursor: pointer;">📄 Download Text Report</button></a>'
                st.markdown(href_text, unsafe_allow_html=True)
    
    else:
        st.warning("No valid data found in selected sessions. Please check file format.")
else:
    st.info("👈 Please upload training session CSV files from the sidebar to begin analysis")
    
    # Display instructions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### How to Use
        1. **Upload CSV Files** from Catapult system
        2. **Select sessions** to analyze
        3. **Explore** player performance metrics
        4. **View** injury risk predictions
        5. **Export** reports for coaching staff
        """)
    
    with col2:
        st.markdown("""
        ### Key Features
        - Individual player performance tracking
        - Multi-session comparison
        - ML-based injury risk prediction
        - Wellness status integration
        - CSV/TXT report generation
        """)
    
    with col3:
        st.markdown("""
        ### Metrics Analyzed
        - Distance covered
        - Player load
        - Maximum velocity
        - Accelerations/decelerations
        - High metabolic load distance
        - Meterage per minute
        """)