import pickle
import sys

def get_categories():
    with open(r'c:\Users\raj18\Desktop\ML\loan aaproval\loan_approval_xgboost_pipeline.pkl', 'rb') as f:
        p = pickle.load(f)
    print("CATEGORIES START")
    cats = p.named_steps['preprocessing'].named_transformers_['cat'].named_steps['encoder'].categories_
    names = p.named_steps['preprocessing'].transformers_[1][2]
    for n, c in zip(names, cats):
        print(f"{n}: {list(c)}")

get_categories()
