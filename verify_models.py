import pickle, os, pandas as pd

# Test ecommerce
try:
    m = pickle.load(open(r'C:\Users\raj18\Desktop\ML\e-commerse\best_model_gradient_boosting.pkl','rb'))
    print('E-commerce model OK:', type(m).__name__)
except Exception as e:
    print('E-commerce FAIL:', e)

# Test loan
try:
    p = pickle.load(open(r'C:\Users\raj18\Desktop\ML\loan aaproval\loan_approval_xgboost_pipeline.pkl','rb'))
    print('Loan pipeline OK:', type(p).__name__)
    test = pd.DataFrame([{
        'person_age':35,'person_income':80000,'person_emp_exp':10,
        'loan_amnt':10000,'loan_int_rate':10.0,'loan_percent_income':0.12,
        'cb_person_cred_hist_length':8,'credit_score':720,
        'person_gender':'male','person_education':'Bachelor',
        'person_home_ownership':'RENT','loan_intent':'EDUCATION',
        'previous_loan_defaults_on_file':'No'
    }])
    pred = p.predict(test)[0]
    prob = p.predict_proba(test)[0][1]
    print(f'Good applicant: {"APPROVED" if pred==1 else "REJECTED"} ({prob:.1%})')
    
    test2 = pd.DataFrame([{
        'person_age':22,'person_income':15000,'person_emp_exp':0,
        'loan_amnt':30000,'loan_int_rate':22.0,'loan_percent_income':0.7,
        'cb_person_cred_hist_length':1,'credit_score':320,
        'person_gender':'female','person_education':'High School',
        'person_home_ownership':'RENT','loan_intent':'PERSONAL',
        'previous_loan_defaults_on_file':'Yes'
    }])
    pred2 = p.predict(test2)[0]
    prob2 = p.predict_proba(test2)[0][1]
    print(f'Bad applicant: {"APPROVED" if pred2==1 else "REJECTED"} ({prob2:.1%})')
except Exception as e:
    print('Loan FAIL:', e)

print('Files in e-commerse:', os.listdir(r'C:\Users\raj18\Desktop\ML\e-commerse'))
print('Files in loan aaproval:', os.listdir(r'C:\Users\raj18\Desktop\ML\loan aaproval'))
