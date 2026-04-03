import kagglehub
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier

def repair_model():
    print("Step 1: Downloading dataset...")
    try:
        path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")
        csv_path = os.path.join(path, "loan_data.csv")
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    print(f"Step 2: Preprocessing data (Shape: {df.shape})...")
    df = df.drop_duplicates()
    df = df.dropna()
    
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    # Define features based on your notebook
    num_columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                   'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                   'credit_score']
    cat_columns = ['person_gender', 'person_education', 'person_home_ownership',
                   'loan_intent', 'previous_loan_defaults_on_file']

    # Modern ColumnTransformer/Pipeline
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_columns),
        ("cat", categorical_transformer, cat_columns)
    ])

    # Replicate model params from notebook
    model = XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=1,
        random_state=42,
        eval_metric="logloss"
    )

    clf = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    print("Step 3: Training model in current environment (scikit-learn 1.8.0)...")
    clf.fit(X, y)

    print("Step 4: Saving repaired model...")
    save_path = r"c:\Users\raj18\Desktop\ML\loan aaproval\loan_approval_xgboost_pipeline.pkl"
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    
    print(f"Model successfully saved to {save_path}")
    print("REPAIR COMPLETE.")

if __name__ == "__main__":
    repair_model()
