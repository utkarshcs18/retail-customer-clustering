# load -> preprocess -> find optimal k -> fit KMeans -> summarize -> visualize

import os
from src.data_preprocessing import preprocess
from src.clustering import compute_elbow_and_silhouette, fit_kmeans, cluster_summary
from src.visualization import plot_elbow, plot_silhouette, plot_clusters_2d, plot_pairwise_clusters

RAW_DATA_PATH = "data/raw/Mall_Customers.csv"
PROCESSED_DATA_PATH = "data/processed/customers_with_clusters.csv"
RESULTS_DIR = "result"


FEATURES = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
INCLUDE_GENDER = False

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    features = ["Gender"] + FEATURES if INCLUDE_GENDER else FEATURES

    # Load + preprocess
    df, df_scaled = preprocess(RAW_DATA_PATH, features, encode_gender=INCLUDE_GENDER)

    X = df_scaled[features]

    # Find optimal k
    inertias, silhouettes = compute_elbow_and_silhouette(X, k_range=range(2, 11))
    plot_elbow(inertias, save_path=f"{RESULTS_DIR}/elbow_plot.png")
    plot_silhouette(silhouettes, save_path=f"{RESULTS_DIR}/silhouette_scores.png")

    print("\nInertia per k:", inertias)
    print("Silhouette score per k:", silhouettes)

    chosen_k = 5

    # Fit final model on the SCALED features
    model, labels = fit_kmeans(X, n_clusters=chosen_k)
    df["Group"] = labels  # store cluster labels on the ORIGINAL (unscaled) df for readable insights

    # Summarize clusters in real-world units (not scaled units)
    summary = cluster_summary(df, features, cluster_col="Group")
    print("\nCluster summary (mean feature values per cluster):")
    print(summary)

    # Save processed data + visualizations
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    plot_clusters_2d(df, "Annual Income (k$)", "Spending Score (1-100)",
                      cluster_col="Group", save_path=f"{RESULTS_DIR}/cluster_visualization.png")
    plot_pairwise_clusters(df, features, cluster_col="Group",
                            save_path=f"{RESULTS_DIR}/cluster_pairplot.png")

    print(f"\nDone. Processed data saved to {PROCESSED_DATA_PATH}, plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
