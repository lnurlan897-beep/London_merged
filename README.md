# 🚲 London Bike Sharing Demand Prediction

A Machine Learning project that predicts the number of bike rentals in London using weather conditions, seasonal information, and time-based features. The model is built with **XGBoost Regressor** and deployed using **Streamlit** for an interactive user experience.

---

## 📌 Project Overview

Bike-sharing systems generate large amounts of data that can be analyzed to understand rental demand patterns. This project uses historical London bike-sharing data to predict the expected number of bike rentals based on:

* Weather conditions
* Temperature
* Humidity
* Wind speed
* Season
* Holiday information
* Weekend information
* Time of day

The goal is to build an accurate regression model that helps estimate bike rental demand.

---

## 🚀 Features

✅ Data preprocessing and feature engineering

✅ Timestamp conversion and extraction of day/hour features

✅ XGBoost Regressor for high-performance prediction

✅ Model serialization using Pickle

✅ Interactive Streamlit web application

✅ User-friendly interface for real-time predictions

---

## 📂 Project Structure

```text
├── app.py                 # Streamlit application
├── model.pkl              # Trained XGBoost model
├── london_merged.csv      # Dataset
├── train_model.py         # Model training script
├── requirements.txt
└── README.md
```

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Streamlit
* Pickle

---

## 📊 Data Preprocessing

The following preprocessing steps were applied:

1. Load the dataset using Pandas.
2. Convert the `timestamp` column to datetime format.
3. Extract:

   * Day
   * Hour
4. Remove the original timestamp column.
5. Split the dataset into training and testing sets.
6. Train an XGBoost Regressor model.

---

## 🤖 Model Training

The model is trained using:

```python
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)
```

Performance is evaluated using the **R² Score**.

---

## 💾 Saving the Model

```python
import pickle

pickle.dump(model, open("model.pkl", "wb"))
```

---

## 🌐 Streamlit Deployment

Run the application locally:

```bash
streamlit run app.py
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/london-bike-sharing-prediction.git
```

Navigate to the project folder:

```bash
cd london-bike-sharing-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🎯 Example Use Cases

* Bike-sharing demand forecasting
* Smart city analytics
* Transportation planning
* Data science portfolio projects
* Machine learning regression studies

---

## 📈 Future Improvements

* Hyperparameter tuning
* Feature importance visualization
* Weather API integration
* Real-time predictions
* Cloud deployment
* Advanced dashboard analytics

---

## 👨‍💻 Author

Developed by **Nurlan Aliyev**

If you found this project useful, consider giving it a ⭐ on GitHub.

---

## 📜 License

This project is licensed under the MIT License.
