import json

file_path = r"C:\Users\raj18\Desktop\ML\Indian_Languages_Audio_Classification.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell["source"]
        text = "".join(source)
        
        # 1. Increase spectrogram detail
        text = text.replace("n_mels=128", "n_mels=256")
        text = text.replace("cmap='viridis'", "cmap='magma'")
        
        # 2. Add extra Dense Layer (1024)
        old_dense = "layers.Dense(512, activation='relu'),\n    layers.BatchNormalization(),\n    layers.Dropout(0.5),\n    layers.Dense(NUM_CLASSES, activation='softmax')"
        new_dense = "layers.Dense(1024, activation='relu'),\n    layers.BatchNormalization(),\n    layers.Dropout(0.5),\n    layers.Dense(512, activation='relu'),\n    layers.BatchNormalization(),\n    layers.Dropout(0.5),\n    layers.Dense(NUM_CLASSES, activation='softmax')"
        text = text.replace(old_dense, new_dense)
        
        # 3. Increase unfrozen layers and reduce LR
        text = text.replace(
            "for layer in base_model.layers[:-50]:  # Unfreeze 50 deeper geometric layers instead of 30",
            "for layer in base_model.layers[:-80]:  # Unfreeze 80 deeper geometric layers instead of 50"
        )
        
        text = text.replace(
            "optimizer=keras.optimizers.Adam(1e-5),",
            "optimizer=keras.optimizers.Adam(5e-6),"
        )
        
        # 4. Increase Epochs
        text = text.replace(
            "epochs=15,",
            "epochs=30,"
        )
        text = text.replace(
            "epochs=20,",
            "epochs=40,"
        )
        text = text.replace(
            "patience=5",
            "patience=8"
        )
        
        # Reconstruct exactly with newlines
        import io
        new_source = []
        # Using string splitlines with True to keep the newline character
        new_source = [line for line in text.splitlines(keepends=True)]
        
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
