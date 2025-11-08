# 🚔 PatrolIQ - Smart Safety Analytics Platform

### 🌐 Domain: Public Safety and Urban Analytics  
**Developed by:** Yogesh Kumar V  
**Tech Stack:** Python | Streamlit | Machine Learning | MLflow | Plotly | Folium | UMAP | HDBSCAN  

---

## 🎯 Project Objective

PatrolIQ is a **Smart Safety Intelligence Platform** designed to analyze massive urban crime datasets using **unsupervised machine learning**.  
The platform identifies **crime hotspots**, discovers **temporal crime patterns**, and visualizes **multi-dimensional insights** to assist law enforcement in **data-driven patrol planning and crime prevention**.

---

## 🧠 Problem Statement

Urban areas face significant challenges in **crime prediction, hotspot identification, and patrol optimization** due to the volume and complexity of crime data.

As a **crime intelligence analyst** at the *Chicago Police Department*, your task is to answer critical operational questions:

- 🧭 *Where should we patrol tonight?*  
- 🕒 *When do most crimes occur?*  
- 🏘️ *Which neighborhoods are high-risk?*

By analyzing **500,000 recent crime records**, PatrolIQ aims to uncover patterns and insights that can **reduce crime and improve urban safety**.

---

## 🧩 Key Skills & Technologies
- 🐍 **Python**, **Streamlit Cloud Deployment**
- 🧮 **Machine Learning & Unsupervised Learning (K-Means, DBSCAN, Hierarchical)**
- 📊 **Dimensionality Reduction (PCA, t-SNE, UMAP)**
- 🧭 **Geospatial Analysis (Folium, GeoPandas)**
- ⚙️ **MLflow Experiment Tracking**
- 📈 **Plotly Interactive Visualizations**
- 🧼 **Data Cleaning, Sampling, and Feature Engineering**

---

## 🧰 Folder Structure

PatrolIQ/
│
├── requirements.txt
├── README.md
├── .gitignore
├── mlruns.db
├── mlflow_server.ps1
│
├── src/
│ ├── data_preprocessing/
│ │ ├── clean_data.py
│ │ └── validate_data.py
│ ├── analysis/
│ │ ├── eda_pipeline.py
│ │ └── feature_engineering.py
│ └── models/
│ ├── dimensionality_reduction.py
│ ├── geo_clustering.py
│ ├── temporal_clustering.py
│ └── utils.py
│ ├── app/
│ ├── pages/
│ │ ├── 01_🎯_Clustering_Analysis.py
│ │ ├── 02_⏰_Temporal_Analysis.py
│ │ ├── 03_🔬_Dimensionality_Reduction.py
│ │ ├── 04_📊_EDA_Insights.py
│ │ ├── 05_🗺️_Geographic_Heatmaps.py
│ │ └── 06_📈_MLflow_Monitoring.py
│ └── 🏠_Home.py
│
├── data/
│ ├── raw/
│ └── processed/
│ ├── model_ready_data.csv
│ └── sample_500000_rows.csv
│
├── models/
├── mlruns/
├── mlartifacts/
└── reports/
├── figures/
└── summaries/

---

## ⚙️ Approach and Workflow

### **Step 1 – Data Acquisition & Preprocessing**
- Downloaded **Chicago Crime Dataset (7.8M records)** from the [Chicago Data Portal](https://data.cityofchicago.org/).
- Sampled **500,000 recent crime records**.
- Cleaned missing values, duplicates, and outliers.
- Extracted **temporal features** (hour, weekday, season, etc.).
- Validated data integrity and structure.

### **Step 2 – Exploratory Data Analysis (EDA)**
- Distribution across **33 crime types**.
- Temporal trends: hourly, daily, and monthly.
- Geographic crime mapping using **Folium heatmaps**.
- Arrest and domestic incident correlations.

### **Step 3 – Feature Engineering**
- Created temporal, geographic, and categorical features.
- Encoded **crime types and locations**.
- Normalized coordinates and derived **crime severity scores**.
- Generated **model-ready dataset** with 22+ features.

### **Step 4 – Clustering Analysis**
**Geographic Crime Hotspot Detection:**
- K-Means → 9 hotspots
- DBSCAN → High-density crime zones
- Hierarchical → Nested area relationships  
Evaluation: *Silhouette score*, *Davies–Bouldin index*

**Temporal Pattern Clustering:**
- K-Means on (Hour, Weekday, Month)
- Identified **3–5 crime time clusters**
- Highlighted **peak hours and high-risk months**

### **Step 5 – Dimensionality Reduction**
- **PCA** → Reduced 22 features → 3 PCs (explaining >70% variance)
- **t-SNE / UMAP** → 2D visualization of clusters
- Identified top features driving crime patterns

### **Step 6 – MLflow Integration**
- Centralized **experiment tracking**
- Logs clustering parameters, metrics, models, and figures
- Enables model comparison and version control

### **Step 7 – Streamlit Application**
- Multi-page web dashboard with:
  - Geographic crime heatmaps
  - Temporal analysis charts
  - Dimensionality reduction visualizations
  - MLflow experiment metrics
- Responsive layout with **Plotly, Folium, and Streamlit UI**

### **Step 8 – Cloud Deployment**
- Deployed on **Streamlit Cloud** using GitHub CI/CD.
- Fully interactive dashboard accessible across devices.

---

## 🧭 Data Flow Diagram
Chicago Crime Dataset (7.8M)
            ↓
Data Cleaning & Sampling (500K)
            ↓
Feature Engineering
            ↓
Clustering (Geographic + Temporal)
            ↓
Dimensionality Reduction (PCA + t-SNE + UMAP)
            ↓
MLflow Experiment Tracking
            ↓
Streamlit App (Interactive Dashboard)
            ↓
Streamlit Cloud Deployment
---


## 📈 Business Use Cases

### 🚓 Police Departments
- Optimize **patrol routes** and **resource allocation**
- Identify **high-risk zones** and **peak times**
- Data-driven decision making for **crime prevention**

### 🏙️ City Administrations
- Enhance **urban planning** and **public safety**
- Justify **budget allocation** and infrastructure needs

### 🧾 Law Enforcement Analytics Firms
- Offer **crime intelligence analytics** as a service
- Develop **predictive policing models**

### 🚑 Emergency Response Systems
- Prioritize calls based on **risk zones**
- Optimize **multi-agency response** and deployment

---

## 🧮 Expected Technical Deliverables

| Module | Expected Result |
|--------|----------------|
| Data Preprocessing | 500K clean, validated records |
| Geographic Clustering | 5–10 distinct hotspots |
| Temporal Clustering | 3–5 time-based patterns |
| Dimensionality Reduction | 70%+ variance explained |
| MLflow | Experiment tracking with metrics |
| Streamlit | Multi-page interactive web app |
| Deployment | Cloud-hosted production app |

---

## 🧑‍💻 Installation & Execution

### Clone the repository
```bash
git clone https://github.com/<your-username>/PatrolIQ.git
cd PatrolIQ
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run MLflow Server (optional)
```bash
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlartifacts
```

### Launch Streamlit App
```bash
streamlit run src/app/🏠_Home.py
```

---

## ☁️ Streamlit Cloud Deployment

The live app is available at:  
🔗 **https://<Yogesh-Venkat>-PatrolIQ.streamlit.app**

Deployed directly from GitHub with **auto-rebuild CI/CD**.

---

## 🏁 Conclusion

PatrolIQ demonstrates how **data-driven crime analysis** can transform public safety management.  
By combining **machine learning, visualization, and automation**, it empowers decision-makers to act faster and smarter — making cities safer, one insight at a time. 🌆✨

> 💡 *“Turning raw data into actionable safety intelligence.”*
