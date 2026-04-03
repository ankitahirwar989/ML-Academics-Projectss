"""
🍎🥕 Fruit & Vegetable Classifier - GUI Application
=====================================================
Usage: python gui_fruitveg.py
"""

import os
import sys
import threading
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

model = None
class_names = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'fruit_veg_classifier_final.keras')
IMG_HEIGHT = 224
IMG_WIDTH = 224

DEFAULT_CLASS_NAMES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage',
    'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn',
    'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes',
    'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango',
    'onion', 'orange', 'paprika', 'pear', 'peas',
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans',
    'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
    'watermelon'
]

# ---------- Colors ----------
BG = "#0d1117"
CARD = "#161b22"
GREEN = "#238636"
BLUE = "#58a6ff"
RED = "#f85149"
WHITE = "#f0f6fc"
GRAY = "#8b949e"
OK = "#3fb950"
DARK = "#21262d"
LINE = "#30363d"


class FruitVegApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit & Vegetable Classifier")
        self.root.geometry("600x700")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        # Center
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 300
        y = (self.root.winfo_screenheight() // 2) - 350
        self.root.geometry(f"600x700+{x}+{y}")

        self.selected_file = None
        self.photo_image = None
        self.model_ready = False

        self.build_ui()
        print("Starting model load...")
        self.root.after(200, self.load_model_async)

    def build_ui(self):
        # ---- Header ----
        tk.Label(self.root, text="🍎🥕 Fruit & Vegetable Classifier",
                 font=("Segoe UI", 18, "bold"), fg=WHITE, bg=GREEN,
                 pady=12).pack(fill='x')

        # ---- Status ----
        self.status_var = tk.StringVar(value="⏳ Loading model... please wait")
        self.status_label = tk.Label(self.root, textvariable=self.status_var,
                                      font=("Segoe UI", 10), fg=GRAY, bg=BG)
        self.status_label.pack(pady=8)

        # ---- Buttons (ALWAYS VISIBLE at top) ----
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=5)

        self.browse_btn = tk.Button(btn_frame, text="📂 Browse Image",
                                     font=("Segoe UI", 11, "bold"),
                                     bg=DARK, fg=WHITE, relief='flat',
                                     padx=20, pady=8, cursor="hand2",
                                     command=self.browse_file)
        self.browse_btn.pack(side='left', padx=8)

        self.predict_btn = tk.Button(btn_frame, text="🎯 Classify",
                                      font=("Segoe UI", 11, "bold"),
                                      bg=GREEN, fg=WHITE, relief='flat',
                                      padx=20, pady=8, cursor="hand2",
                                      command=self.do_predict)
        self.predict_btn.pack(side='left', padx=8)

        # ---- Image Preview ----
        preview_frame = tk.Frame(self.root, bg=CARD, highlightbackground=LINE,
                                  highlightthickness=1)
        preview_frame.pack(padx=30, pady=10, fill='x')

        self.image_label = tk.Label(preview_frame,
                                     text="No image selected\nClick 'Browse Image' to start",
                                     font=("Segoe UI", 10), fg=GRAY, bg=BG,
                                     height=12, pady=10)
        self.image_label.pack(padx=10, pady=10, fill='x')

        # ---- Result Section ----
        result_frame = tk.Frame(self.root, bg=CARD, highlightbackground=LINE,
                                 highlightthickness=1)
        result_frame.pack(padx=30, pady=5, fill='both', expand=True)

        self.result_name = tk.Label(result_frame, text="—",
                                     font=("Segoe UI", 26, "bold"),
                                     fg=BLUE, bg=CARD)
        self.result_name.pack(pady=(15, 0))

        self.result_conf = tk.Label(result_frame, text="Select an image and click Classify",
                                     font=("Segoe UI", 11), fg=GRAY, bg=CARD)
        self.result_conf.pack(pady=(0, 8))

        # Separator
        tk.Frame(result_frame, bg=LINE, height=1).pack(fill='x', padx=20)

        tk.Label(result_frame, text="Top 5 Predictions",
                 font=("Segoe UI", 10, "bold"), fg=GRAY, bg=CARD).pack(pady=(8, 5))

        # Top 5 bars
        self.bars_frame = tk.Frame(result_frame, bg=CARD)
        self.bars_frame.pack(padx=20, pady=(0, 15), fill='both', expand=True)

        self.bar_widgets = []
        for i in range(5):
            row = tk.Frame(self.bars_frame, bg=CARD)
            row.pack(fill='x', pady=2)

            lbl = tk.Label(row, text="—", font=("Segoe UI", 9),
                          fg=WHITE, bg=CARD, width=14, anchor='w')
            lbl.pack(side='left')

            canvas = tk.Canvas(row, height=16, bg=DARK, highlightthickness=0)
            canvas.pack(side='left', fill='x', expand=True, padx=5)

            pct = tk.Label(row, text="", font=("Segoe UI", 8, "bold"),
                          fg=GRAY, bg=CARD, width=7, anchor='e')
            pct.pack(side='right')

            self.bar_widgets.append((lbl, canvas, pct))

    def load_model_async(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        global model, class_names
        try:
            print("Importing tensorflow...")
            from tensorflow.keras.models import load_model as keras_load

            model_path = MODEL_PATH
            if not os.path.exists(model_path):
                h5_path = model_path.replace('.keras', '.h5')
                if os.path.exists(h5_path):
                    model_path = h5_path
                else:
                    msg = f"Model not found in {SCRIPT_DIR}"
                    print(f"ERROR: {msg}")
                    self.root.after(0, lambda: self.status_var.set(f"❌ {msg}"))
                    return

            print(f"Loading model from: {model_path}")
            try:
                model = keras_load(model_path)
            except Exception as load_err:
                print(f"Standard load failed ({load_err}). Attempting fallback architecture rebuild...")
                # Fallback: Rebuild the model architecture manually and load weights
                from tensorflow.keras.applications import MobileNetV2
                from tensorflow.keras import layers, models
                base_model = MobileNetV2(
                    weights=None,  # Do not download imagenet again
                    include_top=False,
                    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
                )
                model = models.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.BatchNormalization(),
                    layers.Dropout(0.4),
                    layers.Dense(256, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.3),
                    layers.Dense(len(DEFAULT_CLASS_NAMES), activation='softmax')
                ])
                model.load_weights(model_path)
            print("Model loaded successfully!")

            num_out = model.output_shape[1]
            if num_out == len(DEFAULT_CLASS_NAMES):
                class_names = DEFAULT_CLASS_NAMES
            else:
                class_names = [f'Class_{i}' for i in range(num_out)]

            self.model_ready = True
            self.root.after(0, lambda: self.status_var.set(f"✅ Model loaded! ({num_out} classes)"))

        except Exception as e:
            err_msg = f"❌ Error: {str(e)[:60]}"
            print(f"ERROR loading model: {e}")
            self.root.after(0, lambda msg=err_msg: self.status_var.set(msg))

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.selected_file = path
            self._show_preview(path)
            fname = os.path.basename(path)
            self.status_var.set(f"✅ Selected: {fname} — Click 'Classify' to predict")

    def _show_preview(self, path):
        try:
            img = Image.open(path).convert('RGB')
            img.thumbnail((220, 220), Image.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo_image, text="", height=0)
        except Exception as e:
            self.image_label.config(text=f"Cannot preview image", image="")

    def do_predict(self):
        if not self.selected_file:
            messagebox.showinfo("No Image", "Please click 'Browse Image' to select an image first.")
            return
        if not self.model_ready:
            messagebox.showinfo("Please Wait",
                                "Model is still loading.\nPlease wait a few seconds and try again.")
            return

        self.predict_btn.config(state='disabled', text="⏳ Working...")
        self.status_var.set("🔄 Classifying...")
        threading.Thread(target=self._predict, daemon=True).start()

    def _predict(self):
        try:
            img = Image.open(self.selected_file).convert('RGB')
            img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            print(f"Predicting: {self.selected_file}")
            predictions = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(predictions[0])
            confidence = predictions[0][pred_idx] * 100
            predicted_name = class_names[pred_idx]
            print(f"Result: {predicted_name} ({confidence:.1f}%)")

            top5_idx = np.argsort(predictions[0])[-5:][::-1]
            top5 = [(class_names[i], predictions[0][i] * 100) for i in top5_idx]

            self.root.after(0, lambda: self._show_result(predicted_name, confidence, top5))

        except Exception as e:
            err_msg = f"❌ Error: {str(e)[:60]}"
            print(f"Prediction error: {e}")
            self.root.after(0, lambda msg=err_msg: self.status_var.set(msg))
            self.root.after(0, lambda: self.predict_btn.config(state='normal', text="🎯 Classify"))

    def _show_result(self, name, confidence, top5):
        self.result_name.config(text=name.upper())

        if confidence > 80:
            color = OK
        elif confidence > 50:
            color = "#d29922"
        else:
            color = RED
        self.result_conf.config(text=f"Confidence: {confidence:.1f}%", fg=color)

        max_prob = top5[0][1] if top5 else 1
        bar_colors = [GREEN, BLUE, "#8b949e", "#6e7681", "#484f58"]

        for i, (lbl, canvas, pct) in enumerate(self.bar_widgets):
            if i < len(top5):
                item_name, prob = top5[i]
                lbl.config(text=item_name.capitalize())
                pct.config(text=f"{prob:.1f}%")

                canvas.delete("all")
                canvas.update_idletasks()
                w = canvas.winfo_width()
                if w > 1:
                    bar_w = max(2, int((prob / max(max_prob, 0.01)) * w))
                    canvas.create_rectangle(0, 0, bar_w, 16, fill=bar_colors[i], outline="")

        self.status_var.set(f"✅ Result: {name.upper()} ({confidence:.1f}%)")
        self.predict_btn.config(state='normal', text="🎯 Classify")


def main():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        h5_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            model_path = h5_path
        else:
            print(f"ERROR: Model file not found!")
            print(f"  Expected: {MODEL_PATH}")
            print(f"  Place fruit_veg_classifier_final.keras in: {SCRIPT_DIR}")
            input("Press Enter to exit...")
            return

    root = tk.Tk()
    app = FruitVegApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
