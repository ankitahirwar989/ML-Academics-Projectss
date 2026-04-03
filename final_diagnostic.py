import os
import pickle
import json

def diagnostic(path):
    print(f"--- Diagnosing: {path} ---")
    if not os.path.exists(path):
        print("Error: File does not exist")
        return
    
    with open(path, 'rb') as f:
        raw = f.read()
    
    print(f"File size: {len(raw)} bytes")
    if len(raw) < 10:
        print("Error: File is too small")
        return

    # Check for binary vs text
    try:
        text = raw.decode('utf-8')
        print("WARNING: File is valid UTF-8 text!")
        print(f"First 100 chars: {text[:100]}")
        if text.strip().startswith('{') or text.strip().startswith('['):
            print("This looks like a JSON file!")
    except UnicodeDecodeError:
        print("File is binary (expected for pickle).")
    
    # Try pickle load with explicitly different protocol?
    try:
        obj = pickle.loads(raw)
        print(f"SUCCESS: pickle.loads worked! Type: {type(obj)}")
    except Exception as e:
        print(f"ERROR: pickle.loads failed: {e}")
        import traceback
        traceback.print_exc()

p = "c:\\Users\\raj18\\Desktop\\ML\\e-commerse\\best_model_gradient_boosting.pkl"
diagnostic(p)
p_encoders = "c:\\Users\\raj18\\Desktop\\ML\\e-commerse\\label_encoders.pkl"
diagnostic(p_encoders)
