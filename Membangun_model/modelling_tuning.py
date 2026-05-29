import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import mlflow
import dagshub

# 1. Inisialisasi DagsHub
REPO_OWNER = 'Ridhoms1' 
REPO_NAME = 'Eksperimen_SML_Ridho-Mas-ud-Said'
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)

def train_and_log_model():
    # 2. Memuat Dataset yang sudah diproses
    data_path = 'namadataset_preprocessing/weather_processed.csv'
    df = pd.read_csv(data_path)
    
    # Memisahkan fitur dan target (Asumsi kolom target bernama 'Weather Type')
    X = df.drop(columns=['Weather Type'])
    y = df['Weather Type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Setup Hyperparameter Tuning menggunakan GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    print("Memulai Hyperparameter Tuning...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Evaluasi model
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 4. Memulai MLflow Run (Manual Logging)
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        print("Mencatat hasil ke DagsHub/MLflow...")
        
        # Log Parameter Terbaik
        mlflow.log_params(best_params)
        
        # Log Metrik
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })
        
        # Log Model
        mlflow.sklearn.log_model(best_model, "random_forest_model")
        
        os.makedirs('artifacts', exist_ok=True)
        
        # Artefak 1: Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        
        # Artefak 2: Feature Importances
        feature_importances = best_model.feature_importances_
        fi_dict = {feature: float(importance) for feature, importance in zip(X.columns, feature_importances)}
        fi_path = "artifacts/feature_importances.json"
        with open(fi_path, 'w') as f:
            json.dump(fi_dict, f, indent=4)
        mlflow.log_artifact(fi_path)
        
        print("Selesai! Buka DagsHub untuk melihat hasilnya.")

if __name__ == "__main__":
    train_and_log_model()