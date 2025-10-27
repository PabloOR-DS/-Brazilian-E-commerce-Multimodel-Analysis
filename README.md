# E-commerce Analytics with Machine Learning  
### Multi-Model Approach to Sales Forecasting, Customer Behavior, and Logistics

This project leverages **ML and NLP** to transform raw e-commerce data into actionable insights, improving sales forecasting, customer behavior understanding, and delivery performance. By integrating **time-series forecasting**, **customer segmentation**, **sentiment analysis**, and **delivery-time prediction**, it delivers measurable operational impact.

## Overview

Using the Olist dataset, this project implements a full ML workflow combining predictive analytics, natural language processing, and optimization techniques. Key achievements include:

- **Delivery time prediction** improved substantially over the system baseline: R² increased from 0.16 to 0.69, meaning the model now explains **53% more variance**, with prediction errors reduced by ~5 days (RMSE ≈ 5.1)  
- Identification of **daily and seasonal order patterns** for operational planning  
- Extraction of **multilingual sentiment insights** from customer reviews to inform strategy  
- A **scalable hybrid ML pipeline** combining tabular models, forecasting, and NLP for end-to-end analytics  

**Key components:**

- Data preprocessing & feature engineering across customers, orders, sellers, and reviews  
- Customer segmentation via K-Means clustering  
- Multilingual sentiment analysis with XLM-RoBERTa  
- Time-series forecasting of order volumes using Prophet  
- Delivery prediction with XGBoost and Optuna tuning  
- Visualization & evaluation with interpretability tools  

## Data

Core datasets:

- Orders, order items, customers, sellers, payments, reviews, geolocation (Olist public datasets: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)   

## Models & Metrics

| Task | Model | Description |
|------|-------|------------|
| Forecasting | Prophet | Daily and monthly order volume predictions with seasonality trends |
| Segmentation | K-Means | Groups customers by purchasing and review behavior |
| Sentiment | XLM-RoBERTa | Multilingual sentiment extraction from reviews |
| Delivery prediction | XGBoost + Optuna | Predicts delivery time with engineered features |

**Metrics:** R² improved from 0.16 (system baseline) to 0.69, RMSE ≈ 5.1 days, MAE, MSE, demonstrating strong improvement over system estimates  

## Workflow

1. Merge raw Olist tables  
2. Build behavioral and logistical features  
3. Forecast orders with Prophet  
4. Segment customers via K-Means  
5. Apply sentiment analysis on reviews  
6. Train and tune XGBoost for delivery prediction  
7. Evaluate with time-aware splits and interpretability  

## Results

- **Delivery time prediction:** R² increased from 0.16 to 0.69, explaining 53% more variance than the baseline and reducing errors by ~5 days (RMSE ≈ 5.1)  
- **Order patterns:** Identified daily and seasonal trends to support operational planning  
- **Customer insights:** Extracted actionable sentiment information from multilingual reviews  
- **Pipeline:** Built a scalable hybrid ML workflow combining NLP, forecasting, and tabular ML  

## Installation

```bash
pip install -r requirements.txt
