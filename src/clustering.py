import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compute_elbow_and_silhouette(X, k_range=range(2, 11)):
    inertias = {}
    silhouettes = {}

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        inertias[k] = model.inertia_
        silhouettes[k] = silhouette_score(X, labels)

    return inertias, silhouettes


def fit_kmeans(X, n_clusters: int, random_state: int = 42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)

    return model, labels


def cluster_summary(df: pd.DataFrame, features: list, cluster_col: str = "Group") -> pd.DataFrame:

    return df.groupby(cluster_col)[features].mean().round(2)
