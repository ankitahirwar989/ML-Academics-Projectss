"""
🍎🥕 Fruit & Vegetable Classifier - Local Prediction Script
=============================================================
Usage:
    python predict.py <image_path>

Example:
    python predict.py apple.jpg
    python predict.py "C:\Users\raj18\Downloads\tomato.png"

Requirements:
    - fruit_veg_classifier_final.keras  (saved model, same folder)
    - pip install tensorflow numpy Pillow matplotlib
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model


# ---- Configuration ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'fruit_veg_classifier_final.keras')
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 36 Fruit & Vegetable class names (alphabetical order as used during training)
CLASS_NAMES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage',
    'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn',
    'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes',
    'jalapeno', 'kiwi', 'lemon', 'lettuce', 'mango',
    'onion', 'orange', 'paprika', 'pear', 'peas',
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans',
    'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip',
    'watermelon'
]


def load_and_preprocess_image(image_path):
    """Load an image and preprocess it for the model."""
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img_resized) / 255.0  # Rescale to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, img


def predict(image_path):
    """Predict the fruit/vegetable in an image."""

    # Check image exists
    if not os.path.exists(image_path):
        print(f'❌ Image not found: {image_path}')
        sys.exit(1)

    # Find model file
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        # Try .h5 format
        h5_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_path):
            model_path = h5_path
        else:
            print(f'❌ Model file not found!')
            print(f'   Expected: {MODEL_PATH}')
            print(f'   Place the .keras file in: {SCRIPT_DIR}')
            sys.exit(1)

    # Load model
    print('🔄 Loading model...')
    model = load_model(model_path)

    # Auto-detect class names from model output
    num_classes = model.output_shape[1]
    if num_classes != len(CLASS_NAMES):
        print(f'⚠️ Model has {num_classes} classes, using generic names.')
        class_names = [f'Class_{i}' for i in range(num_classes)]
    else:
        class_names = CLASS_NAMES

    print('✅ Model loaded!')

    # Preprocess image
    print(f'🖼️ Processing: {os.path.basename(image_path)}')
    img_array, original_img = load_and_preprocess_image(image_path)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    confidence = predictions[0][pred_idx] * 100
    predicted_name = class_names[pred_idx]

    # Display results
    print()
    print('=' * 50)
    print('🎯 PREDICTION RESULT')
    print('=' * 50)
    print(f'   Image:       {os.path.basename(image_path)}')
    print(f'   Predicted:   {predicted_name.upper()}')
    print(f'   Confidence:  {confidence:.2f}%')
    print('=' * 50)

    # Top 5 predictions
    top5_idx = np.argsort(predictions[0])[-5:][::-1]
    print('\n🔝 Top 5 Predictions:')
    for rank, idx in enumerate(top5_idx, 1):
        name = class_names[idx]
        prob = predictions[0][idx] * 100
        bar = '█' * int(prob / 2) + '░' * (50 - int(prob / 2))
        marker = ' ◄' if idx == pred_idx else ''
        print(f'   {rank}. {name:<15} {bar} {prob:6.2f}%{marker}')

    # Show image with prediction (optional, needs display)
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Use interactive backend
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(original_img)
        color = '#2ecc71' if confidence > 70 else '#e74c3c'
        plt.title(f'Predicted: {predicted_name.upper()}\nConfidence: {confidence:.1f}%',
                  fontsize=14, fontweight='bold', color=color)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception:
        # If no display available, skip the plot
        pass

    return predicted_name, confidence


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print('❌ Please provide an image path!\n')
        print(f'   python {os.path.basename(__file__)} <image.jpg>\n')
        print('   Supported formats: .jpg, .jpeg, .png, .bmp, .gif')
        sys.exit(1)

    image_file = sys.argv[1]
    predict(image_file)
