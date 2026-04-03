import pickle
import os

def test_load(path):
    print(f"Testing load: {path}")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            print(f"SUCCESS: Loaded {type(obj)}")
            return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

dir_path = "c:\\Users\\raj18\\Desktop\\ML\\e-commerse"
ok1 = test_load(os.path.join(dir_path, 'best_model_gradient_boosting.pkl.repaired'))
ok2 = test_load(os.path.join(dir_path, 'scaler.pkl.repaired'))
ok3 = test_load(os.path.join(dir_path, 'label_encoders.pkl.repaired'))

if ok1 and ok2 and ok3:
    print("\nALL REPAIRED FILES ARE VALID!")
else:
    print("\nREPAIR FAILED for one or more files.")
