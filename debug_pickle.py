import os
import pickle
import sys

# Add the current directory to path if needed
sys.path.append("c:\\Users\\raj18\\Desktop\\ML")

def test_load():
    dir_path = "c:\\Users\\raj18\\Desktop\\ML\\e-commerse"
    files = [
        'best_model_gradient_boosting.pkl',
        'scaler.pkl',
        'label_encoders.pkl'
    ]
    
    for f in files:
        f_path = os.path.join(dir_path, f)
        print(f"--- Testing {f} ---")
        if not os.path.exists(f_path):
            print(f"FAILED: File does not exist at {f_path}")
            continue
            
        try:
            with open(f_path, 'rb') as pf:
                obj = pickle.load(pf)
                print(f"SUCCESS: Loaded {type(obj)}")
        except Exception as e:
            print(f"ERROR loading {f}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_load()
