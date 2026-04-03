import os

def repair_file(path):
    print(f"Repairing: {path}")
    if not os.path.exists(path):
        print("File not found.")
        return
    
    with open(path, 'rb') as f:
        data = f.read()
    
    # Common corruption: \n (0x0A) converted to \r\n (0x0D 0x0A)
    # This script tries to remove \r (0x0D) only when followed by \n (0x0A)
    # and also checks if stripping ALL \r helps (sometimes entire binary is mangled)
    
    repaired_data = data.replace(b'\r\n', b'\n')
    
    if repaired_data == data:
        print("No \r\n patterns found. Trying to strip ALL \r (0x0D)...")
        repaired_data = data.replace(b'\r', b'')
    
    if repaired_data == data:
        print("No \r bytes found. File is likely corrupted in a different way.")
    else:
        new_path = path + ".repaired"
        with open(new_path, 'wb') as f:
            f.write(repaired_data)
        print(f"Created repaired file: {new_path}")

dir_path = "c:\\Users\\raj18\\Desktop\\ML\\e-commerse"
repair_file(os.path.join(dir_path, 'best_model_gradient_boosting.pkl'))
repair_file(os.path.join(dir_path, 'scaler.pkl'))
repair_file(os.path.join(dir_path, 'label_encoders.pkl'))
