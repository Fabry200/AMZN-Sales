
***

# 🛒 Amazon Sales Analysis & MLOps Pipeline

This project focuses on building an end-to-end Data Science and MLOps pipeline using a synthetic Amazon sales dataset. The goal was to extract actionable business insights through Exploratory Data Analysis (EDA) and customer segmentation, and to build a predictive model for delivery times, managed via an MLOps lifecycle.

## 📊 Project Overview

The analysis covers sales data from **2026/01/01 to 2026/02/18**. The project is divided into three main phases:
1.  **EDA & Business Insights**: Analyzing sales distribution, market concentration, and handling data anomalies.
2.  **Customer Segmentation**: Using unsupervised learning (K-Means) to identify high-value customers.
3.  **Predictive Modeling & MLOps**: Developing a regression model to predict delivery times, using MLflow for experiment tracking and Keras Tuner for hyperparameter optimization.

## 🚀 Key Features & Results

### 1. Exploratory Data Analysis (EDA)
*   **Data Quality**: Identified a critical anomaly where shipping and delivery dates were misaligned, resulting in negative time deltas. The dataset was cleaned to ensure integrity.
*   **Sales Modeling**: Modeled `total_sales` using a **Lognormal distribution**, estimating ~$77M in revenue for the next 1000 transactions.
*   **Pareto Analysis**: Discovered that the top 20% of cities contribute to **48.9%** of the total revenue.

### 2. Customer Segmentation
*   Applied **K-Means Clustering** on customer sales data.
*   Identified **Cluster 2** as the "High-Value" segment, responsible for **60.5%** of the total revenue, providing a clear target for marketing campaigns.

### 3. Predictive Modeling (Time to Arrive)
The objective was to predict `time_to_arrive` (total time between order and delivery).

| Model | MAE (Mean Absolute Error) |
| :--- | :--- |
| Decision Tree Regressor (Baseline) | 8.20 |
| Deep Neural Network (Baseline) | 6.61 |
| **DNN Tuned (Keras Tuner)** | **6.02** |

*   **MLOps**: Used **MLflow** to track experiments, parameters, and metrics.
*   **Optimization**: Used **Keras Tuner (Hyperband)** to search for the optimal architecture (layers, units, activation functions).
*   **Deployment**: The final optimized model was saved locally in `.keras` format for deployment.

## 🛠️ Tech Stack

*   **Language**: Python
*   **Data Manipulation**: Pandas, NumPy
*   **Visualization**: Matplotlib, Seaborn
*   **Machine Learning**: Scikit-learn (DecisionTree, KMeans, StandardScaler)
*   **Deep Learning**: TensorFlow / Keras
*   **Hyperparameter Tuning**: Keras Tuner
*   **MLOps**: MLflow

## ⚙️ Installation & Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas scikit-learn tensorflow keras-tuner mlflow matplotlib seaborn
    ```

3.  **Run the pipeline**:
    Ensure `to_normalize.csv` is in the project directory.
    ```bash
    python main.py
    ```
    *Note: The script will create an `mlruns` folder for tracking and save the final model as `custom_dnn_model.keras`.*

## 📂 Project Structure

```
.
├── main.py                  # Main script containing the pipeline logic
├── to_normalize.csv         # Dataset file
├── custom_dnn_model.keras   # Saved optimized model
├── mlruns/                  # MLflow tracking directory
└── README.md                # Project documentation
```

## 📈 Future Improvements

*   Expand the dataset timeframe to capture seasonality better.
*   Implement cross-validation for more robust evaluation.
*   Deploy the model using a REST API (e.g., FastAPI or Flask).

