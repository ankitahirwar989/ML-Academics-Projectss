# 🎓 Viva Notes — Unified ML Portal

---

## 📌 PART 1: WHAT WE BUILT

### Project Overview
We built a **Unified ML Web Portal** — a Flask-based web application that hosts **5 different machine learning models**, each solving a different real-world problem. Users can visit a browser page, input data or upload a file, and get an AI prediction instantly.

### The 5 Models

| # | Model | Problem Type | Algorithm | Input | Output |
|---|---|---|---|---|---|
| 1 | **E-commerce Churn** | Classification | Gradient Boosting | Customer behaviour (18 features) | Churn / Not Churn |
| 2 | **Loan Approval** | Classification | XGBoost Pipeline | Financial data (13 features) | Approved / Rejected |
| 3 | **Face Mask Detection** | Image Classification | MobileNetV2 (CNN) | Photo of face | Mask / No Mask / Incorrect |
| 4 | **Fruit & Vegetable** | Image Classification | MobileNetV2 (CNN) | Photo of fruit/veg | 36 class labels |
| 5 | **Indian Language Audio** | Audio Classification | CNN + Mel-Spectrogram | .wav / .mp3 audio | Language (10 classes) |

---

### Tech Stack
- **Backend:** Python, Flask (web server)
- **ML Libraries:** scikit-learn, XGBoost, TensorFlow/Keras
- **Data Processing:** pandas, NumPy
- **Audio Processing:** librosa
- **Image Processing:** Pillow (PIL)
- **Frontend:** HTML, CSS (Glassmorphism), Vanilla JavaScript
- **Model Storage:** [.pkl](file:///c:/Users/raj18/Desktop/ML/e-commerse/scaler.pkl) (pickle), [.keras](file:///c:/Users/raj18/Desktop/ML/face%20mask%20detection/face_mask_classifier.keras), [.h5](file:///c:/Users/raj18/Desktop/ML/loan%20aaproval/model.weights.h5) files

---

## 📌 PART 2: MACHINE LEARNING — FROM BASICS

### What is Machine Learning?

Machine Learning (ML) is a branch of Artificial Intelligence where we **train a computer to learn patterns from data** instead of manually programming rules.

> **Example:** Instead of writing `if credit_score > 700: approve_loan`, we show the computer 10,000 past loan decisions and it learns the pattern itself.

### 3 Types of ML

| Type | Description | Example |
|---|---|---|
| **Supervised Learning** | Trained on labelled data (input + correct answer) | Loan approval (we know past results) |
| **Unsupervised Learning** | Finds hidden patterns in unlabelled data | Customer segmentation |
| **Reinforcement Learning** | Agent learns by reward/punishment | Game-playing AI |

> **Our project uses Supervised Learning for all 5 models.**

---

### Key ML Concepts

#### 1. Features & Labels
- **Feature (X):** Input variable. E.g., credit_score, age, income
- **Label (Y):** Output we want to predict. E.g., loan_approved = 0 or 1

#### 2. Training vs Testing
- **Training set:** Data the model learns from (~80%)
- **Test set:** Data used to check if the model generalised (~20%)
- **Why separate?** To avoid the model just "memorising" answers

#### 3. Overfitting vs Underfitting

```
Underfitting: Model too simple → misses patterns → poor accuracy on both sets
Overfitting:  Model too complex → memorises training data → bad on new data
Good fit:     Learns the pattern → works well on new data
```

#### 4. Bias-Variance Tradeoff
- **High Bias** = Underfitting (model makes wrong assumptions)
- **High Variance** = Overfitting (model is too sensitive to training data)
- Goal: Find the sweet spot

#### 5. Cross-Validation
Split training data into `k` folds. Train on k-1, validate on 1. Repeat k times.
Gives a more reliable accuracy estimate. We set `random_state=42` for reproducibility.

---

## 📌 PART 3: ALGORITHMS USED

### Algorithm 1: Gradient Boosting (E-commerce Churn)

**What it is:** An ensemble method that builds many **weak Decision Trees** sequentially. Each new tree corrects the errors of the previous one.

**How it works:**
1. Start with a simple prediction (e.g., predict the average)
2. Calculate errors (residuals)
3. Train a small tree to predict those errors
4. Add this tree to the model (with a small weight = **learning rate**)
5. Repeat 50–100 times (= n_estimators)

**Why it's good:**
- Very accurate on tabular data
- Handles mixed data types well
- Resistant to outliers

**Our code:**
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
```

**Key parameters:**
- `n_estimators=50` → Number of trees
- `learning_rate` → How much each tree contributes (default 0.1)
- `max_depth` → How deep each tree can be

---

### Algorithm 2: XGBoost (Loan Approval)

**What it is:** "Extreme Gradient Boosting" — a supercharged, faster version of Gradient Boosting.

**Why XGBoost is better:**
- Uses parallel processing (faster)
- Built-in regularisation (reduces overfitting)
- Handles missing values automatically
- Widely used in Kaggle competitions

**Our pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])
pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('model', xgb.XGBClassifier(n_estimators=100))
])
pipeline.fit(X, y)
```

**What is a Pipeline?**
A Pipeline chains multiple steps (preprocessing + model) so you never accidentally apply scaling only to training data. Prevents **data leakage**.

---

### Algorithm 3: Decision Tree (base of all above)

**Core idea:** A tree of if-else questions.

```
Is credit_score > 700?
├── Yes → Is income > 50000?
│         ├── Yes → APPROVE
│         └── No  → REJECT
└── No  → REJECT
```

**Splitting criterion:** Gini Impurity or Entropy (Information Gain) — measures how "pure" a split makes the children.

---

### Algorithm 4: Convolutional Neural Network — CNN (Images & Audio)

**Used for:** Face Mask, Fruit/Veg, Audio models

**Why CNN for images?**
Regular neural networks treat every pixel independently — too many weights. CNNs use **filters** that slide over the image and detect local features (edges, shapes).

**Layers in a CNN:**
1. **Conv2D** — Applies filters to detect features (edges → shapes → faces)
2. **MaxPooling2D** — Shrinks the image, keeps strongest features
3. **GlobalAveragePooling2D** — Flattens to a single vector
4. **Dense** — Final classification layer
5. **Softmax** — Outputs probability for each class

**Activation Functions:**
- **ReLU** [f(x) = max(0, x)](file:///c:/Users/raj18/Desktop/ML/app.py#230-233) — Kills negative values, prevents vanishing gradients
- **Softmax** — Converts raw scores to probabilities that sum to 1

---

### Algorithm 5: MobileNetV2 (Transfer Learning)

**What is Transfer Learning?**
Instead of training from scratch (requires millions of images), we use a model **already trained on 1.4 million images** (ImageNet) and fine-tune it for our specific task.

**MobileNetV2:**
- Light, fast architecture designed for mobile devices
- Uses **Depthwise Separable Convolutions** (fewer parameters, same accuracy)
- Input size: 224×224 for fruit/veg and audio; 128×128 for face mask

**Our approach:**
```python
base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
# Add our own classification head on top
model = Sequential([base, GlobalAveragePooling2D(), Dense(256, activation='relu'),
                    Dense(num_classes, activation='softmax')])
```

**Fine-tuning:** We unfreeze the last few layers of MobileNetV2 so they adapt to our data while keeping the earlier layers (basic feature detectors) frozen.

---

### How Audio Classification Works (Mel-Spectrogram)

Audio can't be fed directly into an image CNN. We convert it:

1. Load audio with `librosa.load()`
2. Generate a **Mel-Spectrogram** — a 2D image showing frequency over time
3. Save spectrogram as an image (224×224)
4. Feed into MobileNetV2 like a normal image

**Why Mel scale?** Human hearing is logarithmic. Mel scale compresses high frequencies the same way our ears do, making it more natural for speech/language.

```python
S = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)  # Convert power to decibels
```

---

## 📌 PART 4: LIBRARIES — EXPLAINED

### 1. NumPy
- Numerical computing library
- Provides `ndarray` — fast N-dimensional arrays
- Used for: matrix operations, random data generation
```python
import numpy as np
X = np.array([[1,2],[3,4]])          # Create array
X_scaled = X / 255.0                 # Normalise image pixels
```

### 2. pandas
- Data manipulation library
- Uses `DataFrame` — like an Excel table in Python
```python
import pandas as pd
df = pd.read_csv('data.csv')
df['age'].mean()                      # Column operations
df.drop('id', axis=1)                 # Remove column
df.select_dtypes(include='object')    # Get categorical columns
```

### 3. scikit-learn (sklearn)
- The standard ML library in Python
- Contains: preprocessing, models, evaluation, pipelines

**Key classes we used:**
- `LabelEncoder` — Converts ['Male','Female'] → [0, 1]
- `OneHotEncoder` — Converts category to sparse binary columns
- `StandardScaler` — Scales features to mean=0, std=1 (z-score normalisation)
- `GradientBoostingClassifier` — Our churn model
- `Pipeline` — Chain of steps
- `ColumnTransformer` — Apply different transforms to different columns

**Why scale data?**
Algorithms like SVM and KNN are distance-based — if income (50,000) and age (30) aren't scaled, income dominates. StandardScaler fixes this:
```
z = (x - mean) / std
```

### 4. XGBoost
- Optimised gradient boosting library
- `XGBClassifier` — for binary/multiclass classification
- Key advantage: faster, regularised, handles missing values

### 5. TensorFlow / Keras
- Deep learning framework by Google
- Keras is the high-level API on top of TensorFlow
```python
import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('face_mask_classifier.keras')
predictions = model.predict(image_array)
```

### 6. Flask
- Lightweight Python web framework
- Creates HTTP routes that the browser accesses

```python
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/api/loan/predict', methods=['POST'])
def loan_predict():
    data = request.json           # Get JSON from browser
    prediction = model.predict(data)
    return jsonify({'result': int(prediction)})  # Send back JSON

app.run(debug=True)
```

### 7. librosa
- Audio analysis library
- Used for: loading audio, trimming silence, generating spectrograms
```python
import librosa
y, sr = librosa.load('audio.wav', sr=22050)   # Load audio
y, _ = librosa.effects.trim(y, top_db=20)     # Trim silence
S = librosa.feature.melspectrogram(y=y, sr=sr) # Mel spectrogram
```

### 8. Pillow (PIL)
- Python image library
- Used to load, resize, and convert images for CNN input
```python
from PIL import Image
img = Image.open('photo.jpg').convert('RGB').resize((224, 224))
arr = np.array(img) / 255.0  # Normalise to [0,1]
```

### 9. pickle
- Python's built-in serialisation module
- Converts Python objects (models, scalers) to binary files to save/load
```python
import pickle
pickle.dump(model, open('model.pkl', 'wb'))  # Save
model = pickle.load(open('model.pkl', 'rb')) # Load
```

---

## 📌 PART 5: PREPROCESSING STEPS (FOR TABULAR DATA)

### Step-by-step for E-commerce model:

1. **Load data** → `pd.read_csv()`
2. **Drop irrelevant columns** → e.g., Customer ID
3. **Separate X and y** → features and target
4. **Identify column types:**
   - Categorical: `['PreferredLoginDevice', 'Gender', ...]`
   - Numerical: `['Tenure', 'SatisfactionScore', ...]`
5. **Encode categoricals** → `LabelEncoder().fit_transform()`
6. **Scale numericals** → `StandardScaler().fit_transform()`
7. **Train model** → `.fit(X, y)`
8. **Save** → `pickle.dump()`

> **Important:** We call `.fit_transform()` on training data but only `.transform()` on test/new data to prevent data leakage.

---

## 📌 PART 6: HOW THE FLASK APP WORKS

```
Browser → HTTP Request → Flask Route → Load Model → Predict → JSON Response → Browser displays result
```

1. User opens `http://localhost:5000/loan`
2. Browser fetches `/api/loan/features` → gets field names and dropdown options
3. User fills the form and clicks submit
4. JavaScript sends a **POST** request to `/api/loan/predict` with form data as JSON
5. Flask receives it, loads the XGBoost pipeline, runs [predict()](file:///c:/Users/raj18/Desktop/ML/face%20mask%20detection/gui_facemask.py#237-262) and `predict_proba()`
6. Returns `{'prediction': 1, 'probability': 0.87}` as JSON
7. JavaScript reads the response and shows "APPROVED" with 87% confidence

**Lazy Loading:** We load models on first request (not at server start) to allow the server to boot even if some models are missing:
```python
models_dict = {'loan': None}

def load_loan():
    if not models_dict['loan']:
        models_dict['loan'] = pickle.load(open('pipeline.pkl','rb'))
    return models_dict['loan']
```

---

## 📌 PART 7: EVALUATION METRICS

### For Classification Problems (all our models):

| Metric | Formula | What it means |
|---|---|---|
| **Accuracy** | Correct / Total | % of all predictions that are right |
| **Precision** | TP / (TP+FP) | Of predicted positives, how many are actually positive |
| **Recall** | TP / (TP+FN) | Of actual positives, how many did we catch |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean of precision and recall |
| **AUC-ROC** | Area under ROC curve | Model's ability to distinguish classes (1.0 = perfect) |

> **For loan approval:** Recall matters more — missing an actual defaulter (false negative) is expensive for the bank.

---

## 📌 PART 8: VIVA Q&A

**Q: What is the difference between classification and regression?**
> Classification predicts a **category** (Yes/No, label). Regression predicts a **continuous number** (price, temperature).

**Q: Why did you use Gradient Boosting for churn and not a simple Decision Tree?**
> A single Decision Tree overfits easily. Gradient Boosting builds many trees sequentially, each correcting the previous one's errors, resulting in much better generalisation.

**Q: What is the difference between XGBoost and Gradient Boosting?**
> XGBoost is faster (parallel processing), uses L1/L2 regularisation to prevent overfitting, handles missing values automatically, and has built-in cross-validation support.

**Q: Why do we normalise/scale features?**
> Many algorithms (SVM, KNN, Neural Networks) are sensitive to the scale of inputs. If income (50,000) and age (30) aren't scaled, income will dominate. StandardScaler transforms each feature to have mean=0 and std=1.

**Q: What is Transfer Learning?**
> Reusing a model trained on a large dataset (ImageNet with 1.4M images) for a smaller task. We keep the learned feature detectors and only retrain the final classification layers. Much faster and requires far less data.

**Q: What is a Mel-Spectrogram?**
> A visual representation of audio — time on X-axis, frequency (in Mel scale) on Y-axis, colour = intensity. The Mel scale mimics how human ears perceive pitch (logarithmic). We convert audio clips to spectrogram images and classify them with a CNN.

**Q: What is a Pipeline in sklearn?**
> A Pipeline chains preprocessing steps and the model together into one object. When you call `pipeline.fit()`, it applies all steps in order. Prevents data leakage because scaling parameters learned on training data are correctly applied to test data.

**Q: What is overfitting? How do you prevent it?**
> Overfitting = model memorises training data, fails on new data. Prevention:
> - More training data
> - Regularisation (L1/L2 in XGBoost)
> - Dropout layers (in neural networks)
> - Early stopping
> - Cross-validation

**Q: What is softmax activation?**
> Converts raw output scores (logits) into probabilities that sum to 1. Used in the final layer of multi-class classifiers. E.g., [2.1, 0.3, 1.5] → [0.65, 0.08, 0.27].

**Q: What is the role of Flask in this project?**
> Flask is the web server. It handles HTTP requests from the browser, loads the ML models, runs predictions, and returns results as JSON. It bridges the ML backend with the HTML/CSS/JS frontend.

**Q: What is pickle?**
> Python's built-in serialisation library. It converts any Python object (trained model, scaler, encoder) into a binary byte stream and saves it to a [.pkl](file:///c:/Users/raj18/Desktop/ML/e-commerse/scaler.pkl) file. On restart, `pickle.load()` restores the exact same object.

**Q: What is LabelEncoder vs OneHotEncoder?**
> - **LabelEncoder:** ['Cat', 'Dog', 'Bird'] → [0, 1, 2] (ordinal, implies order — use cautiously)
> - **OneHotEncoder:** ['Cat', 'Dog', 'Bird'] → [[1,0,0], [0,1,0], [0,0,1]] (no implied order, preferred for nominal data)

**Q: What is the difference between `.fit()`, `.transform()`, and `.fit_transform()`?**
> - `.fit()` — Learns parameters (e.g., mean and std for scaler) from the data
> - `.transform()` — Applies learned parameters to new data
> - `.fit_transform()` — Does both in one step (only on training data!)

---

## 📌 PART 9: PROBLEMS WE FACED & HOW WE SOLVED THEM

| Problem | Cause | Solution |
|---|---|---|
| `UnpicklingError: STACK_GLOBAL requires str` | Binary corruption in .pkl files (CRLF characters) | Regenerated fresh model files using venv Python |
| `ModuleNotFoundError: pandas` | pip and python were different environments | Used `python -m pip install` for correct venv |
| `UnicodeDecodeError` on pkl load | joblib trying to open binary pkl as text | Removed joblib, used only stdlib [pickle](file:///c:/Users/raj18/Desktop/ML/app.py#47-52) |
| Loan approval always showing "Rejected" | XGBoost pipeline was corrupted | Regenerated loan model with synthetic realistic data |
| Dropdowns showing "Yes/No" for Gender/Education | Backend cat_mapping failed silently | Hardcoded category values directly in HTML/JS |
| E-commerce "files not found" despite files existing | `BASE_DIR` path mismatch, old code cached in models_dict | Fixed path, cleaned duplicate function definitions |

---

*Good luck in your viva! Remember: explain the WHY behind each choice, not just the WHAT.*
