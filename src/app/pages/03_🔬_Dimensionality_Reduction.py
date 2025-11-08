"""
PatrolIQ - Dimensionality Reduction Visualizations
Interactive PCA, t-SNE, and UMAP visualizations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import joblib
from PIL import Image

# Paths
BASE_DIR = Path(__file__).resolve().parents[3]
PCA_2D = BASE_DIR / "reports" / "summaries" / "pca2_embeddings.csv"
TSNE_2D = BASE_DIR / "reports" / "summaries" / "tsne_embeddings.csv"
UMAP_2D = BASE_DIR / "reports" / "summaries" / "umap_embeddings.csv"
PCA_IMPORTANCE = BASE_DIR / "reports" / "summaries" / "pca_feature_importance.csv"
DIM_SUMMARY = BASE_DIR / "reports" / "summaries" / "dimensionality_summary.json"
VARIANCE_FIG = BASE_DIR / "reports" / "figures" / "pca_variance.png"

st.set_page_config(page_title="Dimensionality Reduction", page_icon="🔬", layout="wide")

st.title("🔬 Dimensionality Reduction & Pattern Discovery")
st.markdown("Explore hidden crime patterns using PCA, t-SNE, and UMAP")

# Load data
@st.cache_data
def load_embeddings():
    embeddings = {}
    if PCA_2D.exists():
        embeddings['pca'] = pd.read_csv(PCA_2D)
    if TSNE_2D.exists():
        embeddings['tsne'] = pd.read_csv(TSNE_2D)
    if UMAP_2D.exists():
        embeddings['umap'] = pd.read_csv(UMAP_2D)
    return embeddings

@st.cache_data
def load_importance():
    if PCA_IMPORTANCE.exists():
        return pd.read_csv(PCA_IMPORTANCE, index_col=0)
    return None

@st.cache_data
def load_summary():
    if DIM_SUMMARY.exists():
        with open(DIM_SUMMARY) as f:
            return json.load(f)
    return None

embeddings = load_embeddings()
importance_df = load_importance()
summary = load_summary()

# Overview metrics
if summary:
    st.subheader("📊 Dimensionality Reduction Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        variance = summary.get('explained_variance', 0)
        st.metric(
            "PCA Explained Variance (3 Components)",
            f"{variance*100:.1f}%",
            help="Percentage of total variance captured by first 3 principal components"
        )
    
    with col2:
        if 'top_features' in summary:
            top_feat = summary['top_features'][0]
            st.metric(
                "Most Important Feature",
                top_feat,
                help="Feature with highest contribution to principal components"
            )

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 PCA Analysis",
    "🎨 t-SNE Visualization",
    "✨ UMAP Projection",
    "🔍 Feature Importance"
])

with tab1:
    st.header("Principal Component Analysis (PCA)")
        # PCA 2D scatter
    if 'pca' in embeddings:
        st.subheader("🔵 PCA 2D Projection")
        
        pca_df = embeddings['pca']
        
        # Determine color column (cluster labels if available)
        color_col = None
        for col in ['KMeans_Geo', 'HDBSCAN_Geo', 'DBSCAN_Geo']:
            if col in pca_df.columns:
                color_col = col
                break
        
        if color_col:
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color=color_col,
                title="PCA 2D: Crime Pattern Visualization",
                labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'},
                opacity=0.6,
                color_continuous_scale='Viridis'
            )
        else:
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                title="PCA 2D: Crime Pattern Visualization",
                labels={'PC1': 'First Principal Component', 'PC2': 'Second Principal Component'},
                opacity=0.6
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 PCA reduces high-dimensional crime data to 2D while preserving maximum variance")
    # Variance plot
    if VARIANCE_FIG.exists():
        st.subheader("📊 Explained Variance")
        st.image(str(VARIANCE_FIG), caption="Cumulative Explained Variance by Principal Components")
        
        if summary:
            st.info(f"✅ First 3 components explain {summary.get('explained_variance', 0)*100:.1f}% of total variance")
    

    else:
        st.warning("⚠️ PCA embeddings not found. Please run dimensionality_reduction.py")

with tab2:
    st.header("t-SNE Visualization")
    
    if 'tsne' in embeddings:
        st.subheader("🎨 t-SNE 2D Projection")
        
        tsne_df = embeddings['tsne']
        
        # Determine color column
        color_col = None
        for col in ['KMeans_Geo', 'HDBSCAN_Geo', 'DBSCAN_Geo']:
            if col in tsne_df.columns:
                color_col = col
                break
        
        if color_col:
            fig = px.scatter(
                tsne_df,
                x='TSNE1',
                y='TSNE2',
                color=color_col,
                title="t-SNE: Crime Pattern Clusters",
                labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
                opacity=0.6,
                color_continuous_scale='Plasma'
            )
        else:
            fig = px.scatter(
                tsne_df,
                x='TSNE1',
                y='TSNE2',
                title="t-SNE: Crime Pattern Clusters",
                labels={'TSNE1': 't-SNE Component 1', 'TSNE2': 't-SNE Component 2'},
                opacity=0.6
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **About t-SNE:**
        - t-SNE (t-Distributed Stochastic Neighbor Embedding) is excellent for visualizing clusters
        - Preserves local structure and reveals natural groupings in data
        - Particularly effective for identifying distinct crime pattern groups
        """)
    else:
        st.warning("⚠️ t-SNE embeddings not found. Please run dimensionality_reduction.py")

with tab3:
    st.header("UMAP Projection")
    
    if 'umap' in embeddings:
        st.subheader("✨ UMAP 2D Projection")
        
        umap_df = embeddings['umap']
        
        # Determine color column
        color_col = None
        for col in ['KMeans_Geo', 'HDBSCAN_Geo', 'DBSCAN_Geo']:
            if col in umap_df.columns:
                color_col = col
                break
        
        if color_col:
            fig = px.scatter(
                umap_df,
                x='UMAP1',
                y='UMAP2',
                color=color_col,
                title="UMAP: Crime Pattern Manifold",
                labels={'UMAP1': 'UMAP Component 1', 'UMAP2': 'UMAP Component 2'},
                opacity=0.6,
                color_continuous_scale='Turbo'
            )
        else:
            fig = px.scatter(
                umap_df,
                x='UMAP1',
                y='UMAP2',
                title="UMAP: Crime Pattern Manifold",
                labels={'UMAP1': 'UMAP Component 1', 'UMAP2': 'UMAP Component 2'},
                opacity=0.6
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **About UMAP:**
        - UMAP (Uniform Manifold Approximation and Projection) balances local and global structure
        - Faster than t-SNE and better at preserving global relationships
        - Reveals both fine-grained clusters and overall data topology
        """)
    else:
        st.warning("⚠️ UMAP embeddings not found. Please run dimensionality_reduction.py")

with tab4:
    st.header("Feature Importance Analysis")
    
    if importance_df is not None:
        st.subheader("🏆 Top Features Contributing to Principal Components")
        
        # Display top features
        top_features = importance_df.nlargest(15, 'importance')
        
        fig = px.bar(
            top_features,
            x='importance',
            y=top_features.index,
            orientation='h',
            title="Top 15 Most Important Features",
            labels={'importance': 'Importance Score', 'y': 'Feature'},
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📋 Feature Loadings on Principal Components")
        
        # Display detailed loadings
        st.dataframe(
            importance_df.style.background_gradient(cmap='coolwarm', axis=0),
            use_container_width=True,
            height=400
        )
        
        st.markdown("""
        **Understanding Feature Importance:**
        - Higher absolute values indicate stronger contribution to principal components
        - PC1 captures the most variance in the data
        - PC2 and PC3 capture orthogonal (independent) variations
        - Features with high loadings across multiple PCs are most informative
        """)
        
        # Summary insights
        st.subheader("💡 Key Insights")
        
        if summary and 'top_features' in summary:
            st.markdown("**Top 5 Most Important Features:**")
            for i, feat in enumerate(summary['top_features'][:5], 1):
                st.write(f"{i}. **{feat}**")
    else:
        st.warning("⚠️ Feature importance data not found. Please run dimensionality_reduction.py")

# Comparison section
st.markdown("---")
st.subheader("🔄 Comparison: PCA vs t-SNE vs UMAP")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **PCA**
    - ✅ Linear transformation
    - ✅ Preserves global structure
    - ✅ Fast computation
    - ✅ Explains variance
    - ❌ May miss non-linear patterns
    """)

with col2:
    st.markdown("""
    **t-SNE**
    - ✅ Non-linear transformation
    - ✅ Excellent cluster separation
    - ✅ Preserves local structure
    - ❌ Slow for large datasets
    - ❌ Non-deterministic
    """)

with col3:
    st.markdown("""
    **UMAP**
    - ✅ Non-linear transformation
    - ✅ Balances local & global
    - ✅ Faster than t-SNE
    - ✅ Deterministic
    - ✅ Better scalability
    """)

st.info("💡 **Recommendation**: Use PCA for initial exploration, t-SNE for cluster visualization, and UMAP for balanced analysis")