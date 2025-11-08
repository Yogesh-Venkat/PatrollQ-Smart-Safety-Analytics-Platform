"""
PatrolIQ - Temporal Pattern Analysis
Interactive temporal visualizations and trend analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "data" / "processed" / "sample_500000_rows.csv"
TEMP_METRICS = BASE_DIR / "reports" / "summaries" / "temporal_clustering_metrics.json"
TEMP_SUMMARY = BASE_DIR / "reports" / "summaries" / "temporal_cluster_summary.csv"

st.set_page_config(page_title="Temporal Analysis", page_icon="⏰", layout="wide")

st.title("⏰ Temporal Pattern Analysis")
st.markdown("Analyze crime patterns across time: hourly, daily, monthly, and seasonal trends")

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Create temporal features if not present
    if 'Year' not in df.columns:
        df['Year'] = df['Date'].dt.year
    if 'Month' not in df.columns:
        df['Month'] = df['Date'].dt.month
    if 'Day' not in df.columns:
        df['Day'] = df['Date'].dt.day
    if 'Hour' not in df.columns:
        df['Hour'] = df['Date'].dt.hour
    if 'Weekday' not in df.columns:
        df['Weekday'] = df['Date'].dt.weekday
    if 'Season' not in df.columns:
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
    
    return df
@st.cache_data
def load_metrics():
    metrics = {}
    if TEMP_METRICS.exists():
        with open(TEMP_METRICS) as f:
            metrics['temp'] = json.load(f)
    return metrics
def load_temporal_summary():
    if TEMP_SUMMARY.exists():
        return pd.read_csv(TEMP_SUMMARY)
    return None
temp_summary = load_temporal_summary()
metrics = load_metrics()
# Load data
with st.spinner("Loading crime data..."):
    df = load_data()



# Tabs
tab1, tab2, tab3, tab4, tab5= st.tabs([
    "⏰ Temporal Clustering",
    "📅 Hourly Patterns", 
    "📆 Daily/Weekly Trends", 
    "📊 Monthly/Seasonal", 
    "🔥 Heatmaps"
])

# ----------------------------- TAB 1 -----------------------------
with tab1:
    st.header("Temporal Crime Pattern Clustering")
    
    if 'temp' in metrics:
        temp_metrics = metrics['temp']
        
        st.subheader("📊 K-Means Temporal Clustering")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of Patterns", temp_metrics['k'])
        with col2:
            if temp_metrics['silhouette'] is not None:
                st.metric("Silhouette Score", f"{temp_metrics['silhouette']:.4f}")
        with col3:
            if temp_metrics['davies_bouldin'] is not None:
                st.metric("Davies-Bouldin Index", f"{temp_metrics['davies_bouldin']:.4f}")
        
        if temp_summary is not None:
            st.subheader("🕐 Temporal Pattern Characteristics")
            
            # Format the summary
            display_summary = temp_summary.copy()
            display_summary.columns = ['Cluster', 'Avg Hour', 'Avg Weekday', 'Avg Month']
            display_summary['Cluster'] = display_summary['Cluster'].apply(lambda x: f"Pattern {x}")
            
            st.dataframe(display_summary, use_container_width=True)
            
            # Visualize patterns
            fig = go.Figure()
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            for i, row in temp_summary.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Hour'], row['Weekday'], row['Month']],
                    theta=['Hour of Day', 'Day of Week', 'Month'],
                    fill='toself',
                    name=f'Pattern {i}',
                    line_color=colors[i % len(colors)]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 24])
                ),
                showlegend=True,
                title="Temporal Pattern Profiles",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Peak times
        if 'peak_hours' in temp_metrics and 'peak_months' in temp_metrics:
            st.subheader("🔥 Peak Crime Times")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Top 5 Peak Hours:**")
                for i, hour in enumerate(temp_metrics['peak_hours'], 1):
                    st.write(f"{i}. {hour}:00")
            
            with col2:
                st.write("**Top 3 Peak Months:**")
                month_names = {1:'January', 2:'February', 3:'March', 4:'April', 
                             5:'May', 6:'June', 7:'July', 8:'August', 
                             9:'September', 10:'October', 11:'November', 12:'December'}
                for i, month in enumerate(temp_metrics['peak_months'], 1):
                    st.write(f"{i}. {month_names.get(month, month)}")
    else:
        st.warning("⚠️ Temporal clustering metrics not found. Please run temporal_clustering.py")

# ----------------------------- TAB 2 -----------------------------
with tab2:
    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Year filter
    years = sorted(df['Year'].unique())
    selected_years = st.sidebar.multiselect("Select Years", years, default=years[-3:])

    # Crime type filter
    crime_types = ['All'] + sorted(df['Primary Type'].unique().tolist())
    selected_crime = st.sidebar.selectbox("Crime Type", crime_types)

    # Apply filters
    filtered_df = df[df['Year'].isin(selected_years)]
    if selected_crime != 'All':
        filtered_df = filtered_df[filtered_df['Primary Type'] == selected_crime]

    st.info(f"📊 Analyzing {len(filtered_df):,} records")

    st.subheader("Hourly Crime Distribution")
    
    hourly = filtered_df['Hour'].value_counts().sort_index()
    
    fig = px.line(
        x=hourly.index,
        y=hourly.values,
        markers=True,
        title="Crime Frequency by Hour of Day"
    )
    fig.update_traces(line_color='#1f77b4', line_width=3)
    fig.update_layout(
        height=500,
        xaxis=dict(title="Hour of Day", dtick=1),
        yaxis=dict(title="Number of Crimes")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Peak hours
    col1, col2, col3 = st.columns(3)
    peak_hours = hourly.nlargest(3)
    
    with col1:
        st.metric("🔴 Peak Hour #1", f"{peak_hours.index[0]}:00", f"{peak_hours.values[0]:,} crimes")
    with col2:
        st.metric("🟠 Peak Hour #2", f"{peak_hours.index[1]}:00", f"{peak_hours.values[1]:,} crimes")
    with col3:
        st.metric("🟡 Peak Hour #3", f"{peak_hours.index[2]}:00", f"{peak_hours.values[2]:,} crimes")

# ----------------------------- TAB 3 -----------------------------
with tab3:
    
    st.subheader("Daily and Weekly Patterns")
    
    # Weekday distribution
    weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    weekday_counts = filtered_df['Weekday'].map(weekday_map).value_counts()
    weekday_counts = weekday_counts.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                             'Friday', 'Saturday', 'Sunday'])
    
    fig = px.bar(
        x=weekday_counts.index,
        y=weekday_counts.values,
        title="Crime Frequency by Day of Week",
        color=weekday_counts.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis=dict(title="Day of Week"),
        yaxis=dict(title="Number of Crimes")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Weekend vs Weekday metrics
    col1, col2 = st.columns(2)
    weekend = filtered_df[filtered_df['Weekday'].isin([5, 6])]
    weekday = filtered_df[~filtered_df['Weekday'].isin([5, 6])]
    
    with col1:
        st.metric("📅 Weekday Crimes", f"{len(weekday):,}")
    with col2:
        st.metric("🎉 Weekend Crimes", f"{len(weekend):,}")

# ----------------------------- TAB 4 -----------------------------
with tab4:
    st.subheader("Monthly and Seasonal Trends")
    
    # Monthly trend
    monthly = filtered_df.groupby(['Year', 'Month']).size().reset_index(name='Count')
    monthly['YearMonth'] = pd.to_datetime(monthly[['Year', 'Month']].assign(DAY=1))
    
    fig = px.line(
        monthly,
        x='YearMonth',
        y='Count',
        title="Monthly Crime Trend Over Time",
        markers=True
    )
    fig.update_layout(
        height=500,
        xaxis=dict(title="Date"),
        yaxis=dict(title="Number of Crimes")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal distribution
    st.subheader("Seasonal Distribution")
    season_counts = filtered_df['Season'].value_counts()
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    season_counts = season_counts.reindex(season_order)
    
    fig = px.bar(
        x=season_counts.index,
        y=season_counts.values,
        title="Crime Frequency by Season",
        color=season_counts.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis=dict(title="Season"),
        yaxis=dict(title="Number of Crimes")
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Season metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("❄️ Winter", f"{season_counts['Winter']:,}")
    with col2:
        st.metric("🌸 Spring", f"{season_counts['Spring']:,}")
    with col3:
        st.metric("☀️ Summer", f"{season_counts['Summer']:,}")
    with col4:
        st.metric("🍂 Fall", f"{season_counts['Fall']:,}")

# ----------------------------- TAB 5 -----------------------------
with tab5:
    st.subheader("Temporal Heatmaps")
    
    # Hour x Weekday heatmap
    heatmap_data = filtered_df.groupby(['Weekday', 'Hour']).size().reset_index(name='Count')
    heatmap_pivot = heatmap_data.pivot(index='Weekday', columns='Hour', values='Count').fillna(0)
    
    # Map weekday numbers to names
    heatmap_pivot.index = heatmap_pivot.index.map(weekday_map)
    heatmap_pivot = heatmap_pivot.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                           'Friday', 'Saturday', 'Sunday'])
    
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour of Day", y="Day of Week", color="Crime Count"),
        title="Crime Heatmap: Day of Week vs Hour",
        aspect="auto",
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("🔥 Darker colors indicate higher crime frequency")

# ----------------------------- SUMMARY -----------------------------
st.markdown("---")
st.subheader("📊 Temporal Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    busiest_hour = filtered_df['Hour'].mode()[0]
    st.metric("⏰ Peak Crime Hour", f"{busiest_hour}:00")

with col2:
    busiest_day = filtered_df['Weekday'].mode()[0]
    st.metric("📅 Peak Crime Day", weekday_map[busiest_day])

with col3:
    busiest_month = filtered_df['Month'].mode()[0]
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                   7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    st.metric("📆 Peak Crime Month", month_names[busiest_month])

with col4:
    busiest_season = filtered_df['Season'].mode()[0]
    st.metric("🌍 Peak Crime Season", busiest_season)
