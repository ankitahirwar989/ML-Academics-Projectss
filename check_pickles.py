import sys
import pickle

def check_pkl(path):
    print(f"Checking {path}...")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            print("Loaded type:", type(obj))
            if hasattr(obj, 'feature_names_in_'):
                print("Features:", list(obj.feature_names_in_))
            elif isinstance(obj, dict):
                print("Keys:", list(obj.keys()))
                for k, v in obj.items():
                    print(f"  {k} classes: {list(v.classes_)}")
            else:
                print("No feature_names_in_ found.")
    except Exception as e:
        print("Error:", e)

check_pkl(r"c:\Users\raj18\Desktop\ML\e-commerse\scaler.pkl")
check_pkl(r"c:\Users\raj18\Desktop\ML\e-commerse\label_encoders.pkl")
check_pkl(r"c:\Users\raj18\Desktop\ML\loan aaproval\scaler.pkl")
check_pkl(r"c:\Users\raj18\Desktop\ML\loan aaproval\label_encoders.pkl")
