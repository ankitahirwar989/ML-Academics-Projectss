import pickle
import pandas as pd
import os
import xgboost

def verify():
    model_path = os.path.join("loan aaproval", "loan_approval_xgboost_pipeline.pkl")
    print(f"Loading model from {model_path}...")
    
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    
    # Create a dummy test sample (Good applicant)
    test_data = {
        'person_age': [30.0],
        'person_income': [100000.0],
        'person_emp_exp': [10.0],
        'loan_amnt': [5000.0],
        'loan_int_rate': [7.5],
        'loan_percent_income': [0.05],
        'cb_person_cred_hist_length': [5.0],
        'credit_score': [750.0],
        'person_gender': ['male'],
        'person_education': ['Bachelor'],
        'person_home_ownership': ['MORTGAGE'],
        'loan_intent': ['EDUCATION'],
        'previous_loan_defaults_on_file': ['No']
    }
    
    # Exact ordering as expected by app.py
    columns = [
        'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'credit_score', 'person_gender', 'person_education', 'person_home_ownership',
        'loan_intent', 'previous_loan_defaults_on_file'
    ]
    
    df = pd.DataFrame(test_data)[columns]
    
    print("Running prediction...")
    pred = pipeline.predict(df)[0]
    prob = pipeline.predict_proba(df)[0][1]
    
    print(f"Result: Prediction={pred}, Probability={prob:.4f}")
    if pred == 0:
        print("Status: APPROVED (Correct)")
    else:
        print("Status: REJECTED")

if __name__ == "__main__":
    verify()
