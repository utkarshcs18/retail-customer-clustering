import matplotlib.pyplot as plt
import seaborn as sns


def plot_elbow(inertias: dict, save_path: str = None):
    plt.figure(figsize=(6, 5))
    plt.plot(list(inertias.keys()), list(inertias.values()), marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_silhouette(silhouettes: dict, save_path: str = None):
    plt.figure(figsize=(6, 5))
    plt.plot(list(silhouettes.keys()), list(silhouettes.values()), marker="o", color="green")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score for Optimal k")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_clusters_2d(df, x_col: str, y_col: str, cluster_col: str = "Group", save_path: str = None):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df, x=x_col, y=y_col, hue=cluster_col, palette="Set2", s=60
    )
    plt.title(f"Customer Segments: {x_col} vs {y_col}")
    plt.grid(True)
    plt.legend(title="Cluster")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_pairwise_clusters(df, features: list, cluster_col: str = "Group", save_path: str = None):
    plot_df = df[features + [cluster_col]]
    g = sns.pairplot(plot_df, hue=cluster_col, palette="Set2")

    if save_path:
        g.savefig(save_path, bbox_inches="tight")
    plt.close()
