import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.joblib")
PIPELINE_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")

def perform_eda(df):
    print("Performing EDA...")
    print(f"Dataset shape: {df.shape}")
    print(df.info())
    print(df.describe())
    print("\nMissing values:\n", df.isnull().sum())

def preprocess_data(df):
    print("Preprocessing data...")
    # Drop customerID
    df = df.drop(columns=['customerID'], errors='ignore')
    
    # Handle TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Separate target
    X = df.drop(columns=['Churn'])
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return X, y

def build_pipeline(X):
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    preprocessor = build_pipeline(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': roc_auc}
        print(f"--- {name} ---")
        print(f"Accuracy: {acc:.4f}  |  Precision: {prec:.4f}  |  Recall: {rec:.4f}  |  F1: {f1:.4f}  |  ROC-AUC: {roc_auc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
            
    print(f"\nBest Model elected: {best_name} (F1 Score: {best_f1:.4f})")
    
    return best_model, preprocessor, best_name, X_train

def save_artifacts(model, preprocessor, feature_names):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PIPELINE_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    print("Model and preprocessor saved to 'models/' directory.")

def plot_feature_importance(model, preprocessor, feature_names, model_name):
    try:
        if model_name in ['Random Forest', 'Gradient Boosting']:
            importances = model.feature_importances_
        elif model_name == 'Logistic Regression':
            importances = np.abs(model.coef_[0])
        else:
            return
            
        # Get feature names from preprocessor
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
        num_features = feature_names.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_features = num_features + list(cat_features)
        
        indices = np.argsort(importances)[::-1][:20] # Top 20
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Top 20 Feature Importances ({model_name})")
        plt.bar(range(20), importances[indices], align="center")
        plt.xticks(range(20), [all_features[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"))
        print("Feature importance plot saved.")
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    perform_eda(df)
    X, y = preprocess_data(df)
    best_model, preprocessor, best_name, X_train = train_and_evaluate(X, y)
    save_artifacts(best_model, preprocessor, X_train)
    plot_feature_importance(best_model, preprocessor, X_train, best_name)
    print("Pipeline Execution Completed.")
