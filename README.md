Drought Prediction Model


This project implements machine learning and deep learning models to predict drought conditions based on meteorological data. The system processes historical weather data, engineers relevant features, trains multiple models, and provides drought risk assessments.

Features
Data Processing: Handles missing values, creates time-based features, and calculates drought indicators

Multiple Models: Implements LSTM, GRU, CNN-LSTM, Random Forest, and Gradient Boosting

Drought Classification: Categorizes drought severity into 5 risk levels

Future Prediction: Forecasts drought conditions for next 7 days

Visualization: Includes time series plots, correlation matrices, and prediction comparisons

Key Results
Model	R² Score	RMSE	MAE
Gradient Boosting	0.9992	0.0081	0.0042
Random Forest	0.9986	0.0107	0.0056
GRU	0.7234	0.1519	0.1032
LSTM	0.7223	0.1522	0.1007
CNN-LSTM	0.6955	0.1594	0.1100
Top 5 Important Features:

SaturationDeficit_Avg

Precipitation

Humidity_Avg

SPI_90

AirTemp_Max

Requirements
Python 3.8+

Libraries:

pandas

numpy

scikit-learn

tensorflow (2.13.1)

matplotlib

seaborn

joblib

Installation
git clone https://github.com/yourusername/drought-prediction.git
cd drought-prediction
pip install -r requirements.txt
Usage
Data Preparation:

Place your meteorological data in data/shymkent_cleaned.csv

Run the preprocessing notebook to generate features

Training Models:
# From notebook:
# Data loading and preprocessing
df = pd.read_csv('shymkent_cleaned.csv')
# ... [preprocessing steps]

# Model training
lstm_model = build_lstm_model((30, 35))
history_lstm = lstm_model.fit(...)
Making Predictions:
# Predict next 7 days
future_predictions = predict_drought(lstm_model, scaler_X, scaler_y, recent_data)
for i, pred in enumerate(future_predictions, 1):
    risk_level = classify_drought_risk(pred)
    print(f"Day {i}: Drought Index = {pred:.3f}, Risk Level = {risk_level}")

drought-prediction/
├── data/                   # Dataset directory
│   └── shymkent_cleaned.csv
├── models/                 # Saved models
│   ├── best_drought_model.h5
│   ├── scaler_X.pkl
│   └── scaler_y.pkl
├── notebooks/              # Jupyter notebooks
│   └── Drought_Prediction.ipynb
├── src/                    # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── prediction.py
├── requirements.txt        # Dependencies
└── README.md
Recommendations
Monitor key parameters: Precipitation, Temperature, Humidity, SPI

Implement early warning system based on predicted drought index

Retrain models quarterly with new data

Combine multiple models in an ensemble approach

Focus on regions showing "Severe" or "Extreme" drought predictions
