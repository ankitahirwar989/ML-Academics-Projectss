import os
import pickle

def check_pkl(path):
    print(f"--- Checking {path} ---")
    if not os.path.exists(path):
        print("File DOES NOT EXIST.")
        return
    
    try:
        with open(path, 'rb') as f:
            header = f.read(20)
            print(f"Header (hex): {header.hex()}")
            print(f"Header (raw): {header}")
            f.seek(0)
            # Try loading with just pickle first
            try:
                obj = pickle.load(f)
                print(f"SUCCESS: Loaded {type(obj)}")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
    except Exception as ex:
        print(f"FAILED to read file: {ex}")

dir_path = "c:\\Users\\raj18\\Desktop\\ML\\e-commerse"
check_pkl(os.path.join(dir_path, 'best_model_gradient_boosting.pkl'))
check_pkl(os.path.join(dir_path, 'label_encoders.pkl'))
check_pkl(os.path.join(dir_path, 'scaler.pkl'))
