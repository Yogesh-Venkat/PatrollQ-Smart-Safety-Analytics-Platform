"""
PatrolIQ - Clustering Analysis Results
Display geographic and temporal clustering results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import streamlit.components.v1 as components
import joblib


# Paths

BASE_DIR = Path(__file__).resolve().parents[3]
GEO_METRICS = BASE_DIR / "reports" / "summaries" / "geo_clustering_metrics.json"
TEMP_METRICS = BASE_DIR / "reports" / "summaries" / "temporal_clustering_metrics.json"
TEMP_SUMMARY = BASE_DIR / "reports" / "summaries" / "temporal_cluster_summary.csv"
CENTERS_PATH = BASE_DIR / "reports" / "summaries" / "kmeans_geo_centers_k9.csv"
MAP_PATH = Path("reports/figures/map_dbscan_geo.html")

st.set_page_config(page_title="Clustering Results", page_icon="🎯", layout="wide")

st.title("🎯 Clustering Analysis Results")
st.markdown("Explore geographic and temporal crime patterns identified by clustering algorithms")

# Load metrics
@st.cache_data
def load_metrics():
    metrics = {}
    if GEO_METRICS.exists():
        with open(GEO_METRICS) as f:
            metrics['geo'] = json.load(f)
    if TEMP_METRICS.exists():
        with open(TEMP_METRICS) as f:
            metrics['temp'] = json.load(f)
    return metrics

@st.cache_data
def load_temporal_summary():
    if TEMP_SUMMARY.exists():
        return pd.read_csv(TEMP_SUMMARY)
    return None

@st.cache_data
def load_cluster_centers():
    if CENTERS_PATH.exists():
        return pd.read_csv(CENTERS_PATH)
    return None

metrics = load_metrics()
temp_summary = load_temporal_summary()
centers = load_cluster_centers()

# Tabs
tab1, tab2,tab3, tab4= st.tabs([
    "📍 K-Means Clustering",
    "🔍 DBSCAN",
    "🌳 Hierarchical Clustering",
    "📊 Performance Metrics"
])

with tab1:

    
    if 'geo' in metrics:
        st.header("📍 K-Means Clustering")
        
        col1, col2 = st.columns(2)
        with col1:
            kmeans_metrics = metrics['geo']['kmeans']
            if kmeans_metrics['silhouette'] is not None:
                st.metric(
                    "Silhouette Score",
                    f"{kmeans_metrics['silhouette']:.4f}",
                    help="Measures how well-separated clusters are (higher is better, range: -1 to 1)"
                )
        with col2:
            if kmeans_metrics['davies_bouldin'] is not None:
                st.metric(
                    "Davies-Bouldin Index",
                    f"{kmeans_metrics['davies_bouldin']:.4f}",
                    help="Measures cluster separation (lower is better)"
                )
        
        if centers is not None:
            
            # Visualize on map
            fig = go.Figure()

            fig.add_trace(go.Scattermapbox(
                lat=centers['Latitude'],
                lon=centers['Longitude'],
                mode='markers+text',
                marker=dict(size=25, color='red', symbol='star'),
                text=[f"H{i}" for i in range(len(centers))],
                textposition="top center",
                textfont=dict(size=14, color='black'),  # ✅ visible on OpenStreetMap
                name='Hotspot Centers',
                hovertemplate='<b>Hotspot %{text}</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
            ))

            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center={"lat": 41.8781, "lon": -87.6298},
                    zoom=10
                ),
                height=600,
                title="9 Geographic Crime Hotspots (K-Means)",
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)


            st.subheader("🎯 Hotspot Center Coordinates")
            centers_display = centers.copy()
            centers_display.index = [f"Hotspot {i}" for i in range(len(centers))]
            st.dataframe(centers_display, use_container_width=True)
        st.markdown("---")

with tab2:
    st.header("🔍 DBSCAN")

    col1, col2 = st.columns(2)
    with col1:
        dbscan_metrics = metrics['geo']['dbscan']
        if dbscan_metrics['silhouette'] is not None:
            st.metric("Silhouette Score", f"{dbscan_metrics['silhouette']:.4f}")
    with col2:
        if dbscan_metrics['davies_bouldin'] is not None:
            st.metric("Davies-Bouldin Index", f"{dbscan_metrics['davies_bouldin']:.4f}")

    st.info("🔍 DBSCAN identifies high-density crime areas and filters out noise/outliers")

    st.markdown("---")

    # ✅ Embed DBSCAN Folium Map
    from pathlib import Path
    import streamlit.components.v1 as components

    MAP_PATH = Path("reports/figures/map_dbscan_geo.html")

    if MAP_PATH.exists():
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            map_html = f.read()
        components.html(map_html, height=600, scrolling=False)
    else:
        st.warning("⚠️ DBSCAN map not found. Please run geo_clustering.py to generate it.")

with tab3:
    st.header("🌳 Hierarchical Clustering")
    
    dendrogram_path = BASE_DIR / "reports" / "figures" / "dendrogram_geo.png"
    if dendrogram_path.exists():
        st.image(str(dendrogram_path), caption="Geographic Crime Zones Dendrogram")
        st.info("📊 The dendrogram shows hierarchical relationships between geographic crime zones")
    else:
        st.warning("⚠️ Dendrogram not found. Please run geo_clustering.py")



with tab4:
    st.header("📊 Clustering Performance Metrics")
    

    if 'geo' in metrics or 'temp' in metrics:
        st.subheader("📈 Comparison Chart")
        
        comparison_data = []
        
        if 'geo' in metrics:
            for algo, vals in metrics['geo'].items():
                if vals['silhouette'] is not None:
                    comparison_data.append({
                        'Algorithm': f"Geo {algo.upper()}",
                        'Silhouette': vals['silhouette'],
                        'Davies-Bouldin': vals['davies_bouldin'] if vals['davies_bouldin'] else 0
                    })
        
        if 'temp' in metrics and metrics['temp']['silhouette'] is not None:
            comparison_data.append({
                'Algorithm': 'Temporal KMeans',
                'Silhouette': metrics['temp']['silhouette'],
                'Davies-Bouldin': metrics['temp']['davies_bouldin'] if metrics['temp']['davies_bouldin'] else 0
            })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df_comp,
                    x='Algorithm',
                    y='Silhouette',
                    title="Silhouette Score Comparison",
                    color='Silhouette',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    df_comp,
                    x='Algorithm',
                    y='Davies-Bouldin',
                    title="Davies-Bouldin Index Comparison",
                    color='Davies-Bouldin',
                    color_continuous_scale='Reds_r'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No clustering metrics available")
    st.markdown("""
    ### Understanding the Metrics
    
    **Silhouette Score**
    - Range: -1 to 1
    - Higher is better
    - Measures how similar an object is to its own cluster compared to other clusters
    - > 0.5: Good clustering
    - 0.2 - 0.5: Acceptable clustering
    - < 0.2: Poor clustering
    
    **Davies-Bouldin Index**
    - Range: 0 to ∞
    - Lower is better
    - Measures the average similarity between clusters
    - < 1.0: Good clustering
    - 1.0 - 2.0: Acceptable clustering
    - > 2.0: Poor clustering
    """)
    
# Summary
st.markdown("---")
st.subheader("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Geographic Clustering:**
    - ✅ Identified 9 distinct crime hotspots using K-Means
    - ✅ DBSCAN detected high-density crime areas
    - ✅ Hierarchical clustering revealed zone relationships
    """)

with col2:
    st.markdown("""
    **Temporal Clustering:**
    - ✅ Discovered 4 distinct time-based crime patterns
    - ✅ Identified peak crime hours and months
    - ✅ Grouped similar temporal behaviors
    """)