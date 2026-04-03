"""
🎙️ Indian Languages Audio Classifier - Advanced 2D CNN (V2)
==========================================================
Uses MobileNetV2 architecture with Mel-Spectrogram Image 
conversion (replacing the old 1D MFCC neural net).
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import threading
import json
import warnings
import io

# Suppress all annoying TensorFlow hardware warnings in the console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Force numpy float format fixes for librosa compatibility warnings
warnings.filterwarnings('ignore')

# Delayed imports
np = None
librosa = None
AudioSegment = None
model = None
idx_to_label = None

# --- UI CONSTANTS ---
BG_COLOR = "#0f172a"
CARD_BG = "#1e293b"
TEXT_COLOR = "#f8fafc"
TEXT_MUTED = "#94a3b8"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#10b981"
HIGHLIGHT = "#0ea5e9"
BAR_BG = "#334155"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'indian_language_classifier.keras')
LABELS_PATH = os.path.join(SCRIPT_DIR, 'label_mapping.json')


class AudioClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Indian Language AI (Spectroscopic 2D CNN)")
        self.root.geometry("850x600")
        self.root.configure(bg=BG_COLOR)
        
        self.selected_file = None
        self.bar_widgets = []
        self.spectrogram_img = None
        self.waveform_img = None
        
        self.file_label = None
        self.browse_btn = None
        self.predict_btn = None
        self.preview_canvas = None
        self.result_language = None
        self.result_confidence = None
        self.bars_frame = None
        self.status_var = None

        self._build_ui()
        threading.Thread(target=self._initialize_backend, daemon=True).start()

    def _initialize_backend(self):
        self.root.after(0, lambda: self.status_var.set("⏳ Loading advanced AI engines (Keras, Librosa, Matplotlib)..."))
        
        global np, librosa, AudioSegment
        import numpy as _np
        import librosa as _librosa
        import librosa.display as _librosa_display
        from pydub import AudioSegment as _AudioSegment
        
        np = _np
        librosa = _librosa
        AudioSegment = _AudioSegment
        
        self.root.after(0, lambda: self.status_var.set("⏳ Loading MobileNetV2 Keras Model..."))
        self._load_model()

    def _load_model(self):
        global model, idx_to_label
        try:
            from tensorflow.keras.models import load_model

            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file missing: {MODEL_PATH}")
            if not os.path.exists(LABELS_PATH):
                raise FileNotFoundError(f"Label map missing: {LABELS_PATH}")

            try:
                model = load_model(MODEL_PATH)
            except Exception:
                # Fallback manual reconstruction if TF versions clash
                import tensorflow as tf
                from tensorflow.keras.applications import MobileNetV2
                from tensorflow.keras import layers
                import h5py

                base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
                model = tf.keras.Sequential([
                    base_model,
                    layers.GlobalAveragePooling2D(),
                    layers.BatchNormalization(),
                    layers.Dropout(0.4),
                    layers.Dense(512, activation='relu'),
                    layers.BatchNormalization(),
                    layers.Dropout(0.5),
                    layers.Dense(10, activation='softmax') # Assuming 10 languages
                ])
                h5_file = h5py.File(MODEL_PATH, 'r')
                if 'model_weights' in h5_file:
                    model.load_weights(MODEL_PATH, by_name=True)
                else:
                    model = load_model(MODEL_PATH, compile=False)

            with open(LABELS_PATH, 'r') as f:
                label_data = json.load(f)
            idx_to_label = label_data['index_to_label']

            self.root.after(0, self._create_bars)
            self.root.after(0, lambda: self.status_var.set("✅ AI Brain Ready. Please select an audio file."))
            self.root.after(0, lambda: self.predict_btn.config(state='normal'))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"❌ Initialization Error: {str(e)[:70]}"))

    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG_COLOR, pady=15)
        header.pack(fill='x')
        tk.Label(header, text="V2 Spectroscopic Language AI", font=("Segoe UI", 20, "bold"), fg=TEXT_COLOR, bg=BG_COLOR).pack()
        tk.Label(header, text="Analyzes voice structure via Mel-Spectrogram Images (MobileNetV2)", font=("Segoe UI", 10), fg=TEXT_MUTED, bg=BG_COLOR).pack()

        # Split layout
        main_frame = tk.Frame(self.root, bg=BG_COLOR, padx=20)
        main_frame.pack(fill='both', expand=True)

        left_col = tk.Frame(main_frame, bg=BG_COLOR, width=400)
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))

        right_col = tk.Frame(main_frame, bg=BG_COLOR, width=350)
        right_col.pack(side='right', fill='y')

        # Control Panel
        control_card = tk.Frame(left_col, bg=CARD_BG, padx=20, pady=20)
        control_card.pack(fill='x', pady=(0, 15))

        self.file_label = tk.Label(control_card, text="No file selected", font=("Segoe UI", 10), fg=TEXT_COLOR, bg=CARD_BG, wraplength=350)
        self.file_label.pack(pady=(0, 10))

        btn_frame = tk.Frame(control_card, bg=CARD_BG)
        btn_frame.pack(fill='x')

        self.browse_btn = tk.Button(btn_frame, text="📁 Browse Audio...", font=("Segoe UI", 10, "bold"),
                                   bg=ACCENT_BLUE, fg="white", relief="flat", padx=15, pady=8, cursor="hand2",
                                   command=self.browse_file)
        self.browse_btn.pack(side='left', expand=True, fill='x', padx=(0, 5))

        self.predict_btn = tk.Button(btn_frame, text="🎯 Predict Language", font=("Segoe UI", 10, "bold"),
                                    bg=ACCENT_GREEN, fg="white", relief="flat", padx=15, pady=8, cursor="hand2",
                                    state='disabled', command=self.predict_async)
        self.predict_btn.pack(side='right', expand=True, fill='x', padx=(5, 0))

        # Visualizer Panel
        vis_card = tk.Frame(left_col, bg=CARD_BG, padx=10, pady=10)
        vis_card.pack(fill='both', expand=True)
        tk.Label(vis_card, text="Spectrogram Vision (What AI Sees)", font=("Segoe UI", 10, "bold"), fg=TEXT_MUTED, bg=CARD_BG).pack(anchor='w', pady=(0, 10))
        
        self.preview_canvas = tk.Canvas(vis_card, bg="#000000", highlightthickness=0, height=140)
        self.preview_canvas.pack(fill='both', expand=True)
        self.preview_canvas.create_text(200, 70, text="Audio vision will appear here...", fill=TEXT_MUTED, font=("Segoe UI", 9))

        # Result Panel (Right)
        result_card = tk.Frame(right_col, bg=CARD_BG, padx=20, pady=20)
        result_card.pack(fill='both', expand=True)

        tk.Label(result_card, text="RESULT", font=("Segoe UI", 12, "bold"), fg=TEXT_MUTED, bg=CARD_BG).pack(pady=(0, 10))
        
        self.result_language = tk.Label(result_card, text="—", font=("Segoe UI", 28, "bold"), fg=HIGHLIGHT, bg=CARD_BG)
        self.result_language.pack(pady=(0, 5))
        
        self.result_confidence = tk.Label(result_card, text="Confidence: —", font=("Segoe UI", 11), fg=TEXT_COLOR, bg=CARD_BG)
        self.result_confidence.pack(pady=(0, 10))

        # Probability Bars
        self.bars_frame = tk.Frame(result_card, bg=CARD_BG)
        self.bars_frame.pack(fill='both', expand=True)

        # Status Bar
        self.status_var = tk.StringVar(value="⏳ Initializing system...")
        tk.Label(self.root, textvariable=self.status_var, font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_COLOR, anchor='w').pack(fill='x', padx=20, pady=5)

    def _create_bars(self):
        if idx_to_label is None:
            return
        
        for w in self.bars_frame.winfo_children():
            w.destroy()
        self.bar_widgets.clear()

        for i in range(len(idx_to_label)):
            lang = idx_to_label[str(i)]
            row = tk.Frame(self.bars_frame, bg=CARD_BG)
            row.pack(fill='x', pady=2)

            lbl = tk.Label(row, text=lang.upper(), font=("Segoe UI", 8, "bold"), fg=TEXT_COLOR, bg=CARD_BG, width=10, anchor='e')
            lbl.pack(side='left', padx=(0, 5))

            canvas = tk.Canvas(row, height=14, bg=BAR_BG, highlightthickness=0)
            canvas.pack(side='left', fill='x', expand=True, padx=(0, 5))

            pct_lbl = tk.Label(row, text="0%", font=("Segoe UI", 8), fg=TEXT_MUTED, bg=CARD_BG, width=5)
            pct_lbl.pack(side='right')

            self.bar_widgets.append((lbl, canvas, pct_lbl))

    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=(("Audio Files", "*.mp3 *.wav *.m4a *.aac *.flac *.wma *.ogg"), ("All Files", "*.*"))
        )
        if path:
            self.selected_file = path
            self.file_label.config(text=f"Selected: {os.path.basename(path)}")
            # Reset UI
            self.result_language.config(text="—")
            self.result_confidence.config(text="Confidence: —")
            for lbl, canvas, pct_lbl in self.bar_widgets:
                canvas.delete("all")
                pct_lbl.config(text="0%")
                lbl.config(fg=TEXT_COLOR)

    def predict_async(self):
        if not self.selected_file:
            messagebox.showwarning("No File", "Please select an audio file first!")
            return
        
        self.predict_btn.config(state='disabled', text="⏳ Analyzing...")
        self.status_var.set("⏳ Generating 2D Spectrogram Phase...")
        threading.Thread(target=self._predict, daemon=True).start()

    def _generate_spectrogram_image(self, audio_array, sr):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import librosa.display
        
        # Spectrogram parameters
        S = librosa.feature.melspectrogram(y=audio_array, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot headless cleanly in thread
        fig = Figure(figsize=(3.5, 3.5), dpi=64) # 224x224 physical pixels
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.axis('off')
        librosa.display.specshow(S_dB, sr=sr, ax=ax, fmax=8000, cmap='viridis')
        
        # Extract RGB directly without saving file
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        
        image = Image.open(buf).convert('RGB')
        image = image.resize((224, 224))
        
        # Update UI visually with the generated image
        disp_width = self.preview_canvas.winfo_width()
        disp_height = self.preview_canvas.winfo_height()
        ui_img = image.resize((disp_width, disp_height), Image.Resampling.LANCZOS)
        
        tk_img = ImageTk.PhotoImage(ui_img)
        self.root.after(0, lambda: self._update_preview(tk_img))
        
        # Map to MobileNet V2 model input format
        img_array = np.array(image, dtype=np.float32) / 255.0
        return img_array

    def _update_preview(self, tk_img):
        self.spectrogram_img = tk_img # Store ref
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=tk_img)

    def _predict(self):
        try:
            file_path = self.selected_file
            
            # Phase 1: Robust Audio Loading
            y = None
            sr = 22050
            
            try:
                self.root.after(0, lambda: self.status_var.set("⏳ Loading Audio File (Librosa)..."))
                y, sr = librosa.load(file_path, sr=sr, duration=15)
            except Exception:
                self.root.after(0, lambda: self.status_var.set("⏳ Librosa failed. Recovering with FFmpeg/PyDub..."))
                try:
                    audio = AudioSegment.from_file(file_path)
                    buffer = io.BytesIO()
                    audio.export(buffer, format='wav')
                    buffer.seek(0)
                    y, sr = librosa.load(buffer, sr=sr, duration=15)
                except Exception as ex:
                    raise RuntimeError(f"FFmpeg missing or file corrupt: {str(ex)}")
            
            if y is None or len(y) == 0:
                raise ValueError("Audio file is completely empty or could not be decoded.")

            # Phase 2: Trim pure silence
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Keep only the first 5 seconds
            y = y[:sr * 5]
            
            if len(y) < sr * 0.5: # Failsafe if audio was entirely static 
                raise ValueError("After removing leading silence, the audio file is completely empty.")

            # Phase 3: Spectrogram Conversion
            self.root.after(0, lambda: self.status_var.set("⏳ Translating Sound into Vision (224x224 Spectrogram)..."))
            img_array = self._generate_spectrogram_image(y, sr)
            X_input = img_array[np.newaxis, ...] # Shape (1, 224, 224, 3)

            # Phase 4: Predict
            self.root.after(0, lambda: self.status_var.set("⏳ Neural Network Processing..."))
            
            try:
                predictions = model.predict(X_input, verbose=0)[0]
            except Exception as e:
                err_msg = str(e).lower()
                if "sequential.call" in err_msg or "incompatible" in err_msg or "shape" in err_msg:
                    raise ValueError(
                        "ARCHITECTURE MISMATCH! The .keras file in your folder is an older version (V1 1D-CNN or V2 128x128). "
                        "Please go to Colab, open the NEWEST V3 Notebook, click 'Run All', and download the latest 224x224 model!"
                    )
                else:
                    raise e
            
            print(f"DEBUG LAYER: {predictions}")
            
            if np.isnan(predictions).any() or np.sum(predictions) == 0:
                raise ValueError("CNN returned Invalid Mathematical Probabilities (Possible Corrupt Model)")
                
            pred_idx = np.argmax(predictions)
            confidence = predictions[pred_idx] * 100
            predicted_lang = idx_to_label[str(pred_idx)]

            # UI Update
            self.root.after(0, lambda: self._show_result(predicted_lang, confidence, predictions))
            self.root.after(0, lambda: self.status_var.set(f"✅ Scanning Complete: Successfully analyzed {len(y)/sr:.1f}s segment!"))
            self.root.after(0, lambda: self.predict_btn.config(state='normal', text="🎯 Predict Language"))
            
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"❌ Error: {str(e)[:80]}"))
            self.root.after(0, lambda: self.predict_btn.config(state='normal', text="🎯 Predict Language"))

    def _show_result(self, language, confidence, probabilities):
        self.result_language.config(text=language.upper())
        self.result_confidence.config(text=f"Confidence: {confidence:.2f}%")

        max_prob = max(probabilities)
        pred_idx = np.argmax(probabilities)

        for i, (lbl, canvas, pct_lbl) in enumerate(self.bar_widgets):
            prob = probabilities[i] * 100
            
            # Animate colors
            if i == pred_idx:
                color = HIGHLIGHT
                lbl.config(fg=HIGHLIGHT, font=("Segoe UI", 8, "bold"))
                pct_lbl.config(fg=HIGHLIGHT, font=("Segoe UI", 8, "bold"))
            else:
                color = "#475569"
                lbl.config(fg=TEXT_COLOR, font=("Segoe UI", 8))
                pct_lbl.config(fg=TEXT_MUTED, font=("Segoe UI", 8))

            pct_lbl.config(text=f"{prob:.1f}%")

            canvas.delete("all")
            canvas.update_idletasks()
            w = canvas.winfo_width()
            
            if w > 1:
                bar_width = int((prob / 100.0) * w)  # Scale exactly by 100% since it's softmax
                canvas.create_rectangle(0, 0, bar_width, 14, fill=color, outline="")

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioClassifierApp(root)
    root.mainloop()
