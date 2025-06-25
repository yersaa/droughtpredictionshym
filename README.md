

---

# 🌾 Drought Prediction Model

This project implements machine learning and deep learning models to predict drought conditions based on meteorological data. It processes historical weather data, engineers features, trains multiple models, and provides drought risk assessments.

---

## 📌 Features

* **Data Processing**: Handles missing values, creates time-based features, calculates drought indicators (e.g., SPI)
* **Multiple Models**: Implements `LSTM`, `GRU`, `CNN-LSTM`, `Random Forest`, and `Gradient Boosting`
* **Drought Classification**: Categorizes drought severity into 5 risk levels
* **Future Prediction**: Forecasts drought conditions for the next 7 days
* **Visualization**: Time series plots, correlation matrices, and model comparison charts

---

## 📊 Key Results

| Model             | R² Score   | RMSE   | MAE    |
| ----------------- | ---------- | ------ | ------ |
| Gradient Boosting | **0.9992** | 0.0081 | 0.0042 |
| Random Forest     | 0.9986     | 0.0107 | 0.0056 |
| GRU               | 0.7234     | 0.1519 | 0.1032 |
| LSTM              | 0.7223     | 0.1522 | 0.1007 |
| CNN-LSTM          | 0.6955     | 0.1594 | 0.1100 |

### 🔝 Top 5 Important Features

1. `SaturationDeficit_Avg`
2. `Precipitation`
3. `Humidity_Avg`
4. `SPI_90`
5. `AirTemp_Max`

---

## ⚙️ Requirements

* Python 3.8+

**Libraries:**

```
pandas  
numpy  
scikit-learn  
tensorflow==2.13.1  
matplotlib  
seaborn  
joblib
```

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/drought-prediction.git
cd drought-prediction
pip install -r requirements.txt
```

---

## 🧪 Usage

### 1. Data Preparation

Place your meteorological data at:

```
data/shymkent_cleaned.csv
```

Run the preprocessing notebook to generate features.

### 2. Training Models

```python
# Load and preprocess data
df = pd.read_csv('shymkent_cleaned.csv')
# ... [apply preprocessing steps]

# Train model
lstm_model = build_lstm_model((30, 35))
history_lstm = lstm_model.fit(...)
```

### 3. Making Predictions

```python
# Predict next 7 days
future_predictions = predict_drought(lstm_model, scaler_X, scaler_y, recent_data)

# Print risk levels
for i, pred in enumerate(future_predictions, 1):
    risk_level = classify_drought_risk(pred)
    print(f"Day {i}: Drought Index = {pred:.3f}, Risk Level = {risk_level}")
```

---

## 📁 Project Structure

```
drought-prediction/
├── data/                  
│   └── shymkent_cleaned.csv
├── models/                 
│   ├── best_drought_model.h5
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
├── notebooks/             
│   └── Drought_Prediction.ipynb
├── src/                   
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
├── requirements.txt        
└── README.md
```

---

## ✅ Recommendations

* Monitor key parameters: `Precipitation`, `Temperature`, `Humidity`, `SPI`
* Implement early warning systems based on predicted drought index
* Retrain models quarterly with updated data
* Use ensemble methods for improved accuracy
* Prioritize regions with "Severe" or "Extreme" drought forecasts

---

Let me know if you want a version with badges (e.g., Python version, license, etc.) or a GitHub Pages version for documentation.
