import pickle, warnings
warnings.filterwarnings('ignore')
from sklearn.utils.validation import check_is_fitted

pipeline = pickle.load(open('loan aaproval/loan_approval_xgboost_pipeline.pkl', 'rb'))

preproc = pipeline.named_steps['preprocessing']
for name, t, cols in preproc.transformers:
    if name == 'cat':
        print(f"Cat OHE handle_unknown: {t.__dict__.get('handle_unknown', 'N/A')}")
        print(f"Cat OHE categories (param): {t.__dict__.get('categories', 'N/A')}")
        print(f"Cat OHE drop: {t.__dict__.get('drop', 'N/A')}")
        print()
        # Check if fitted
        try:
            check_is_fitted(t)
            print("OHE IS fitted")
        except Exception as e:
            print(f"OHE fit check: {e}")
        
        # Try to transform a sample
        import pandas as pd
        sample = pd.DataFrame([['male', 'Bachelor', 'RENT', 'EDUCATION', 'No']], 
                              columns=cols)
        try:
            result = t.transform(sample)
            print(f"Transform worked: shape={result.shape}")
        except Exception as e:
            print(f"Transform error: {e}")
        
        # Try 'female'  
        sample2 = pd.DataFrame([['female', 'Master', 'OWN', 'MEDICAL', 'Yes']], 
                               columns=cols)
        try:
            result2 = t.transform(sample2)
            print(f"Transform2 worked: shape={result2.shape}")
        except Exception as e:
            print(f"Transform2 error: {e}")

# Now try predict the app way - exactly replicating app.py
import pandas as pd
cat_list = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
           'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
           'credit_score', 'person_gender', 'person_education', 'person_home_ownership',
           'loan_intent', 'previous_loan_defaults_on_file']

req = {
    'person_age': '30', 'person_income': '75000', 'person_emp_exp': '5',
    'loan_amnt': '10000', 'loan_int_rate': '10.5', 'loan_percent_income': '0.15',
    'cb_person_cred_hist_length': '5', 'credit_score': '680',
    'person_gender': 'male', 'person_education': 'Bachelor',
    'person_home_ownership': 'RENT', 'loan_intent': 'EDUCATION',
    'previous_loan_defaults_on_file': 'No'
}

input_data = {col: [req.get(col, 0 if col not in cat_list else '')] for col in columns}
df = pd.DataFrame(input_data)

print()
print("Full pipeline predict test:")
try:
    pred = int(pipeline.predict(df)[0])
    prob = float(pipeline.predict_proba(df)[0][1])
    print(f"pred={pred}, prob={prob:.4f} -> {'APPROVED' if pred==1 else 'REJECTED'}")
except Exception as e:
    print(f"PIPELINE ERROR: {e}")
    import traceback; traceback.print_exc()
