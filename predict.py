import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def predict(age, cholesterol, restingbp, max_hr, oldpeak, bmi, riskfactor):
    data = np.array([[age, cholesterol, restingbp, max_hr, oldpeak, bmi, riskfactor]])
    data = scaler.transform(data)
    return model.predict(data)[0]