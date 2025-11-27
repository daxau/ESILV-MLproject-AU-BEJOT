# ESILV-MLproject-AU-BEJOT  
Machine Learning Project – Stock Price Direction Forecasting

##  1. Introduction

This project aims to forecast **next-day stock price direction** (up or down) using:
- Historical stock returns  
- Macroeconomic indicators (interest rates, CPI, GDP, unemployment, etc.)

The goal is to provide a **decision-support tool** for investors, analysts, and students who want to:
- Track stock market behavior  
- Analyze relationships between macro variables and stock movements  
- Predict whether a stock’s next-day return will be **positive or negative**

---

##  Datasets Used

### **Market Data**
- **Yahoo Finance (via `yfinance`)**
  - Daily prices for AAPL, MSFT, CAC40, S&P500, and other global equities  
  - Clean and easy to integrate  
- **Quandl** (optional alternative)

### **Macroeconomic Data**
- **FRED – Federal Reserve Economic Data**
  - Interest rates  
  - Inflation / CPI  
  - Unemployment  
  - GDP growth  
- **World Bank Open Data**

### **Notes on Data Sources**
- `yfinance`: simplified access to global daily stock prices  
- Quandl: rich catalog, often mirrors FRED/OECD indicators  
- FRED: authoritative source for macroeconomic variables  

---

##  Techniques Implemented

The project includes several machine learning models:

- **Logistic Regression**  
- **Random Forest**  
- **XGBoost**  
- **LSTM (Long-Short Term Memory)** for sequential patterns  

### **Evaluation Metrics**
- Accuracy  
- ROC-AUC  
- Precision / Recall  
- Confusion Matrix  

---

##  2. Installation & Usage

The project is structured into Jupyter Notebooks that must be executed **in sequence**.

### **Notebook Pipeline**

| Step | Notebook | Purpose |
|------|----------|---------|
| **1** | `step1_ML_Dataset.ipynb` | Retrieve raw market & macro data; generates `Full_dataset_reference.csv ` |
| **2** | `Step2_ML_EDA_Dataset.ipynb` | Run Exploratory Data Analysis (EDA) on raw dataset; generates `Cleaned_Features_for_ML.csv ` |
| **3** | `Step3_ML_FeatureSelection_KBest.ipynb` | Feature selection based on ANOVA method - alternative dataset, generates `Cleaned_Features_for_ML_20ANOVA.csv ` |
| **4** | `Step4_ML_ModelClassification.ipynb` | Model training & classification experiments |
| **5** | `Step5_ML_Model_ARIMA_8D_forecast.ipynb` | Model training & classification experiments introduced ARIMA model forecast and 8-day rolling direction forecast |
| **6** | `Step6_ML_Ultimate_test.ipynb` | Advanced Model training & classification experiments - ARIMA +20 features +LSTM +8Day rolling forecast |
| **5** | `dashboard.py` | Visualization Dashboard and results interpretation |

---

##  3. Repository Structure

The entire development workflow is organized in the repository **ESILV-MLproject-AU-BEJOT**.

### **Branches**
- **`main`** → Stable consolidated code  
- **`Implementation-Dax-AU`** → Development branch for contributor A  
- **`Implementation-Teofil-BEJOT`** → Development branch for contributor B  

---

##  Folders Overview

