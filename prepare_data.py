import os
import urllib.request
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DATA_DIR = "data"
RAW_DATA_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print("Attempting to download Telco Customer Churn dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, RAW_DATA_PATH)
        print(f"Successfully downloaded to {RAW_DATA_PATH}")
    except Exception as e:
        print(f"Failed to download from IBM repo: {e}")
        print("Synthesizing a realistic dataset instead...")
        synthesize_data()

def synthesize_data():
    np.random.seed(42)
    n_samples = 7043
    
    # Generate Synthetic Data similar to Telco Churn
    data = {
        "customerID": [f"{i:04d}-{np.random.choice(list('ABCDEF'))}{np.random.choice(list('XYZW'))}" for i in range(n_samples)],
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples, p=[0.3, 0.7]),
        "tenure": np.random.randint(0, 73, n_samples),
        "PhoneService": np.random.choice(["Yes", "No"], n_samples, p=[0.9, 0.1]),
        "MultipleLines": np.random.choice(["No phone service", "No", "Yes"], n_samples, p=[0.1, 0.45, 0.45]),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]),
        "OnlineSecurity": np.random.choice(["No", "Yes", "No internet service"], n_samples, p=[0.5, 0.28, 0.22]),
        "OnlineBackup": np.random.choice(["No", "Yes", "No internet service"], n_samples, p=[0.44, 0.34, 0.22]),
        "DeviceProtection": np.random.choice(["No", "Yes", "No internet service"], n_samples, p=[0.44, 0.34, 0.22]),
        "TechSupport": np.random.choice(["No", "Yes", "No internet service"], n_samples, p=[0.5, 0.28, 0.22]),
        "StreamingTV": np.random.choice(["No", "Yes", "No internet service"], n_samples, p=[0.4, 0.38, 0.22]),
        "StreamingMovies": np.random.choice(["No", "Yes", "No internet service"], n_samples, p=[0.4, 0.38, 0.22]),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples, p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples, p=[0.6, 0.4]),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n_samples),
        "MonthlyCharges": np.round(np.random.uniform(18.25, 118.75, n_samples), 2),
    }

    df = pd.DataFrame(data)
    
    # Logic for TotalCharges based on MonthlyCharges and tenure
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).astype(str)
    # Introduce some logical empty strings for tenure=0 to mimic the real dataset
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    
    # Introduce logical rules for Churn to make the ML task meaningful
    # Higher churn probability for Month-to-month, Fiber optic, shorter tenure, higher monthly charges
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(df["Contract"] == "Month-to-month", 0.3, 0.0)
    churn_prob += np.where(df["InternetService"] == "Fiber optic", 0.15, 0.0)
    churn_prob += np.where(df["tenure"] < 12, 0.2, 0.0)
    churn_prob += np.where(df["MonthlyCharges"] > 70, 0.1, 0.0)
    churn_prob -= np.where(df["TechSupport"] == "Yes", 0.1, 0.0)
    churn_prob -= np.where(df["OnlineSecurity"] == "Yes", 0.1, 0.0)
    
    # Clip prob to [0,1] range
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    
    df["Churn"] = [np.random.choice(["Yes", "No"], p=[p, 1-p]) for p in churn_prob]
    
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Synthesized dataset saved to {RAW_DATA_PATH}")

if __name__ == "__main__":
    download_data()
