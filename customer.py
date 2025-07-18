import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide",
    page_icon="favicon.png"   # <-- add this line
)

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("🧩 Customer Segmentation Dashboard with K-Means")

# Sidebar: Upload dataset
st.sidebar.header("1️⃣ Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### 📄 Dataset Preview", data.head())

    # Sidebar: Feature selection
    st.sidebar.header("2️⃣ Select Features for Clustering")
    features = st.sidebar.multiselect(
        "Select at least two numerical features:",
        options=data.select_dtypes(include=['float64', 'int64']).columns,
        default=data.select_dtypes(include=['float64', 'int64']).columns[:2]
    )

    if len(features) >= 2:
        X = data[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Sidebar: Elbow plot
        st.sidebar.header("3️⃣ Choose Number of Clusters")
        show_elbow = st.sidebar.checkbox("Show Elbow Plot to find optimal k")
        if show_elbow:
            inertia = []
            K_range = range(1, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_scaled)
                inertia.append(kmeans.inertia_)
            fig_elbow, ax = plt.subplots()
            ax.plot(K_range, inertia, marker='o')
            ax.set_xlabel('Number of clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('Elbow Method')
            st.sidebar.pyplot(fig_elbow)

        # Select k
        k = st.sidebar.slider("Number of clusters (k):", min_value=2, max_value=10, value=3)

        # Train KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        data['Cluster'] = clusters

        # Dashboard: Metrics
        st.subheader("📊 Dashboard Metrics")
        cluster_counts = data['Cluster'].value_counts().sort_index()
        st.write("#### Number of customers per cluster")
        st.bar_chart(cluster_counts)

        st.write("#### Mean of features per cluster")
        st.dataframe(data.groupby('Cluster')[features].mean())

        # PCA visualization when >2 features
        st.subheader("🎨 Cluster Visualization")
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
            df_pca['Cluster'] = clusters
            fig_pca, ax = plt.subplots()
            sns.scatterplot(data=df_pca, x='PC1', y='PC2',
                            hue='Cluster', palette='tab10', s=80, alpha=0.7, ax=ax)
            ax.set_title(f"PCA Visualization ({k} Clusters)")
            st.pyplot(fig_pca)
        else:
            fig_2d, ax = plt.subplots()
            sns.scatterplot(data=data, x=features[0], y=features[1],
                            hue='Cluster', palette='tab10', s=80, alpha=0.7, ax=ax)
            ax.set_title(f"Customer Segmentation ({k} Clusters)")
            st.pyplot(fig_2d)

        # Export data
        st.subheader("⬇️ Export Clustered Data")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='clustered_customers.csv',
            mime='text/csv'
        )

    else:
        st.warning("⚠️ Please select at least two numerical features to cluster.")

else:
    st.info("💡 Upload a CSV file to get started.")

