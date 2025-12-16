import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Credit_Risk_Task_5")

    # 1. Load Data
    data_path = os.path.join('data', 'processed', 'processed_data.csv')
    df = pd.read_csv(data_path)
    
    # 2. Preparation
    target_col = 'FraudResult'
    cols_to_drop = [target_col, 'TransactionId', 'BatchId', 'AccountId', 
                    'SubscriptionId', 'CustomerId', 'TransactionStartTime']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    
    X = pd.get_dummies(df.drop(columns=existing_drops), drop_first=True)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Hyperparameter Tuning (Grid Search)
    model_params = {
        "Tuned_Logistic": {
            "model": LogisticRegression(max_iter=1000, solver='liblinear'),
            "params": {'C': [0.1, 1, 10]}
        },
        "Tuned_RF": {
            "model": RandomForestClassifier(),
            "params": {'n_estimators': [50, 100]}
        }
    }

    for name, config in model_params.items():
        with mlflow.start_run(run_name=name):
            print(f"Searching best parameters for {name}...")
            # GridSearchCV helps satisfy the 'Hyperparameter Tuning' requirement
            grid = GridSearchCV(config['model'], config['params'], cv=3, scoring='roc_auc')
            grid.fit(X_train, y_train)
            
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            # 4. Comprehensive Model Evaluation
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }

            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, name)
            print(f"SUCCESS: {name} logged with all metrics.")

if __name__ == "__main__":
    print("--- Starting Tuned Training ---")
    train_and_evaluate()
    print("--- All Done ---")