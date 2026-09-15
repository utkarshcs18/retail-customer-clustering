# Customer Segmentation using K-Means Clustering

## 📌 Problem Statement
Retail and e-commerce businesses collect customer data (age, annual income, spending behavior) but often fail to use it effectively for targeted marketing. Treating all customers the same leads to inefficient marketing spend and missed opportunities for personalized offers.

This project uses **unsupervised learning (K-Means Clustering)** to segment mall/retail customers into distinct groups based on their **age, annual income, and spending score**. The goal is to identify meaningful customer segments (e.g., high-income low-spenders, young high-spenders) so a business can design targeted marketing strategies for each group.

## 🎯 Objectives
- Perform exploratory data analysis (EDA) on customer data
- Determine the optimal number of clusters using the **Elbow Method** and **Silhouette Score**
- Apply K-Means clustering to segment customers
- Visualize and interpret each segment in business terms
- Provide actionable recommendations per segment

## 🗂️ Folder Structure
```
customer-segmentation-kmeans/
│
├── data/
│   ├── raw/
│   │   └── Mall_Customers.csv
│   └── processed/
│       └── customers_with_clusters.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_optimal_clusters.ipynb
│   └── 03_kmeans_clustering.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── clustering.py
│   └── visualization.py
│
├── results/
│   ├── elbow_plot.png
│   ├── silhouette_scores.png
│   └── cluster_visualization.png
│
├── requirements.txt
├── README.md
└── main.py
```

## 🛠️ Tech Stack
- Python
- Pandas, NumPy (data handling)
- Matplotlib, Seaborn (visualization)
- Scikit-learn (K-Means, Silhouette Score)

## 📊 Dataset
[Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) — contains CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100).

## ⚙️ How to Run
```bash
git clone https://github.com/<your-username>/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
pip install -r requirements.txt
python main.py
```

## 📈 Results
| Cluster | Age Group | Income Level | Spending Score | Segment Label |
|---------|-----------|---------------|-----------------|----------------|
| 0       | -         | -             | -               | *(fill after running)* |
| 1       | -         | -             | -               | *(fill after running)* |
| 2       | -         | -             | -               | *(fill after running)* |

*(Replace with your actual cluster insights once the notebook is run — recruiters look for this table more than the code.)*

## 💡 Key Insights
- *(e.g., "Cluster 2 represents high-income, low-spending customers — a strong target for loyalty/reward campaigns.")*
- *(e.g., "Cluster 4 represents young, high-spending customers — ideal for trend-based marketing.")*

## 🚀 Future Improvements
- Try DBSCAN or Hierarchical Clustering for comparison
- Add more features (purchase frequency, product category) if available
- Deploy as a simple Streamlit dashboard for interactive exploration

## 👤 Author
*(Your name / GitHub / LinkedIn link)*