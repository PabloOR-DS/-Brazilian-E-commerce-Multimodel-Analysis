# E-commerce Analytics with Machine Learning: A Multi-Model Approach to Sales Forecasting, Customer Behavior, and Logistics
### Connecting transaction records to actionable insights on sales performance, purchasing patterns, and delivery processes

The project integrates **time-series forecasting**, **customer segmentation**, **sentiment analysis**, and **delivery-time prediction** into a unified data-driven pipeline for operational optimization.

## Overview

This project builds a full **ML workflow** using real e-commerce data from the Olist dataset.  
It combines predictive analytics, NLP, and optimization techniques to improve forecasting accuracy and delivery performance.

**Key components:**
- **Data preprocessing** and feature engineering across customer, order, seller, and review datasets.  
- **Customer segmentation** using K-Means to identify behavioral patterns.  
- **Sentiment analysis** with multilingual transformer (XLM-RoBERTa).  
- **Time-series forecasting** of order volume using Prophet.  
- **Delivery time prediction** with XGBoost regression and Optuna hyperparameter tuning.  
- **Visualization and evaluation** with detailed metrics and interpretability tools.

## Data

The analysis is based on Olist public datasets, including:
- `olist_orders_dataset.csv`
- `olist_order_items_dataset.csv`
- `olist_customers_dataset.csv`
- `olist_sellers_dataset.csv`
- `olist_order_reviews_dataset.csv`
- `olist_order_payments_dataset.csv`
- `olist_geolocation_dataset.csv`

Additional processed files:
- `reviews_data.csv` (preprocessed reviews with sentiment)
- `forecast_df_plt.csv` (Prophet forecasts for order volumes)

## Models and Methods

| Task | Model / Method | Description |
|------|----------------|-------------|
| Forecasting | **Prophet** | Predicts daily and monthly order volumes with seasonality trends. |
| Segmentation | **K-Means** | Groups customers by purchasing and review behavior. |
| Sentiment | **XLM-RoBERTa** | Multilingual transformer for customer review sentiment extraction. |
| Delivery prediction | **XGBoost + Optuna** | Predicts delivery time using engineered and behavioral features. |

**Metrics:**  
R² (≈0.78 train / 0.69 test), RMSE ≈ 5.1 days, MAE, MSE, and baseline comparison against system estimates.

## Workflow

1. **Load and merge** raw Olist tables.  
2. **Build features** for customer behavior, distance, and estimated delivery performance.  
3. **Train Prophet model** for order volume forecasting.  
4. **Segment customers** with K-Means clustering.  
5. **Run XLM-R sentiment pipeline** on reviews and integrate features.  
6. **Train and tune XGBoost model** for hybrid delivery prediction.  
7. **Evaluate** using time-aware split and model explainability.

## Results

- Improved delivery time prediction accuracy over system estimates.  
- Demonstrated daily and seasonal patterns in order activity.  
- Extracted multilingual sentiment insights from customer reviews.  
- Created scalable hybrid pipeline integrating NLP, forecasting, and tabular ML.

## Installation
To set up the environment and install required dependencies, run:

```bash
pip install -r requirements.txt
