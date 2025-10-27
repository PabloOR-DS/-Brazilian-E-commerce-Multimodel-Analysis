# E-commerce Analytics with Machine Learning  
### Multi-Model Approach to Sales Forecasting, Customer Behavior, and Logistics

This project leverages **ML and NLP** to turn raw e-commerce data into actionable insights for sales forecasting, customer behavior, and delivery optimization. It integrates **time-series forecasting**, **customer segmentation**, **sentiment analysis**, and **delivery-time prediction** into a unified pipeline.

## Overview

Using the Olist dataset, the project implements a complete ML workflow combining predictive analytics, natural language processing, and optimization techniques. Key components:

- **Data preprocessing & feature engineering** across customers, orders, sellers, and reviews  
- **Customer segmentation** via K-Means clustering  
- **Multilingual sentiment analysis** with XLM-RoBERTa  
- **Time-series forecasting** of order volumes using Prophet  
- **Delivery prediction** with XGBoost and Optuna tuning  
- **Visualization & evaluation** with interpretability tools  

## Data

Core datasets:

- Orders, order items, customers, sellers, payments, reviews, geolocation (Olist public datasets)  
- `reviews_data.csv` — preprocessed reviews with sentiment  
- `forecast_df_plt.csv` — Prophet forecasts for order volumes  

## Models & Metrics

| Task | Model | Description |
|------|-------|------------|
| Forecasting | Prophet | Daily and monthly order volume predictions with seasonality trends |
| Segmentation | K-Means | Groups customers by purchasing and review behavior |
| Sentiment | XLM-RoBERTa | Multilingual sentiment extraction from reviews |
| Delivery prediction | XGBoost + Optuna | Predicts delivery time with engineered features |

**Metrics:** R² ≈ 0.78 (train) / 0.69 (test), RMSE ≈ 5.1 days, MAE, MSE, outperforming system estimates

## Workflow

1. Merge raw Olist tables  
2. Build behavioral and logistical features  
3. Forecast orders with Prophet  
4. Segment customers via K-Means  
5. Apply sentiment analysis on reviews  
6. Train and tune XGBoost for delivery prediction  
7. Evaluate with time-aware splits and interpretability  

## Results

- Improved delivery predictions over baseline estimates  
- Identified daily and seasonal order patterns  
- Extracted actionable insights from multilingual reviews  
- Demonstrated a scalable hybrid ML pipeline  

## Installation

```bash
pip install -r requirements.txt
