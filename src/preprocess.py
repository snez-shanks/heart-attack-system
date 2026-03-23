import pandas as pd

def load_data(path):
    # Your file is tab-separated
    return pd.read_csv(path, sep='\t')

def split_data(df):
    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    # Use only useful features
    X = df[['age', 'cholesterol', 'restingbp', 'max_heart_rate', 'oldpeak', 'bmi', 'riskfactor']]
    
    # Target column
    y = df['heartdisease']

    return X, y