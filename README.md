# Heart Attack Prediction System

## Project Overview
The Heart Attack Prediction System is a **machine learning-based application** that predicts the risk of heart disease in individuals based on medical data such as age, cholesterol, blood pressure, and more. This project provides a **user-friendly interface** using **Streamlit**, allowing users to input their health data and receive a risk assessment instantly.

---

## Features
- Predicts the likelihood of a heart attack based on user inputs.
- Interactive **Streamlit web interface** for easy data entry.
- Uses **Machine Learning** models (Random Forest / Logistic Regression) for accurate predictions.
- Visualizes data trends using **Matplotlib** and **Seaborn** (optional).

---

## Technologies Used
- **Python 3.x**  
- **Streamlit** – For web interface  
- **Pandas & NumPy** – For data handling and computations  
- **Scikit-learn** – For ML model training and prediction  
- **Matplotlib & Seaborn** – For data visualization  

---

## Folder Structure
heart-attack-project/
│
├── app.py # Streamlit application file
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── data/ # Dataset CSV files
│ └── heart_updated.csv
└── models/ # Saved ML models
└── heart_model.pkl


---

## Setup Instructions
1. **Clone the repository**
```bash
git clone <your-github-repo-link>
cd heart-attack-project