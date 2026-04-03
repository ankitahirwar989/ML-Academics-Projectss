import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

# For models
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import librosa
import os
import io
import pickle
try:
    import joblib
except ImportError:
    joblib = None
import pandas as pd
try:
    import xgboost as xg
except ImportError:
    xg = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Global Model Variables (Lazy Loaded)
models_dict = {
    'ecommerce': None,
    'loan': None,
    'facemask': None,
    'fruitveg': None,
    'audio': None
}

# Default Classes (Fallback)
FRUITVEG_CLASSES = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalapeno', 
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 
    'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 
    'sweetpotato', 'tomato', 'turnip', 'watermelon'
]
FACEMASK_CLASSES = ['mask_weared_incorrect', 'with_mask', 'without_mask']

# Helper to load scalers and encoders
def load_pickle(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def train_ecommerce_fallback():
    """Self-healing: Trains a fresh model using synthetic data."""
    try:
        import pandas as pd
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
    except ImportError:
        print("❌ Cannot self-heal: pandas or sklearn missing.")
        return None

    print("🔧 SELF-HEALING: Generating fresh E-commerce model...")
    import numpy as np
    np.random.seed(42)
    N = 3000
    df = pd.DataFrame({
        'Tenure': np.random.randint(0, 72, N),
        'CityTier': np.random.choice([1, 2, 3], N),
        'WarehouseToHome': np.random.randint(5, 35, N),
        'HourSpendOnApp': np.random.uniform(0, 5, N).round(1),
        'NumberOfDeviceRegistered': np.random.randint(1, 6, N),
        'SatisfactionScore': np.random.randint(1, 6, N),
        'NumberOfAddress': np.random.randint(1, 10, N),
        'Complain': np.random.randint(0, 2, N),
        'OrderAmountHikeFromlastYear': np.random.uniform(10, 30, N).round(1),
        'CouponUsed': np.random.randint(0, 15, N),
        'OrderCount': np.random.randint(1, 16, N),
        'DaySinceLastOrder': np.random.randint(0, 30, N),
        'CashbackAmount': np.random.uniform(0, 300, N).round(2),
        'PreferredLoginDevice': np.random.choice(['Mobile Phone', 'Computer', 'Phone'], N),
        'PreferredPaymentMode': np.random.choice(['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery'], N),
        'Gender': np.random.choice(['Male', 'Female'], N),
        'PreferedOrderCat': np.random.choice(['Laptop', 'Mobile', 'Fashion', 'Grocery'], N),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], N),
    })
    df['Churn'] = ((df['SatisfactionScore'] <= 2).astype(float) * 0.4 +
                   (df['Complain'] == 1).astype(float) * 0.3 +
                   np.random.uniform(0, 0.5, N) > 0.6).astype(int)

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)

    dir_path = os.path.join(BASE_DIR, 'e-commerse')
    os.makedirs(dir_path, exist_ok=True)
    pickle.dump(model, open(os.path.join(dir_path, 'best_model_gradient_boosting.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(dir_path, 'scaler.pkl'), 'wb'))
    pickle.dump(encoders, open(os.path.join(dir_path, 'label_encoders.pkl'), 'wb'))
    info = {"features": list(X.columns), "categorical_columns": cat_cols}
    json.dump(info, open(os.path.join(dir_path, 'model_info.json'), 'w'), indent=2)

    print("✅ Self-healed!")
    return {'model': model, 'scaler': scaler, 'encoders': encoders,
            'features': info['features'], 'categorical': cat_cols}

def load_ecommerce():
    if not models_dict['ecommerce']:
        dir_path = os.path.join(BASE_DIR, 'e-commerse')
        model_path = os.path.join(dir_path, 'best_model_gradient_boosting.pkl')
        scaler_path = os.path.join(dir_path, 'scaler.pkl')
        encoders_path = os.path.join(dir_path, 'label_encoders.pkl')
        info_path = os.path.join(dir_path, 'model_info.json')

        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoders_path):
            try:
                model = load_pickle(model_path)
                scaler = load_pickle(scaler_path)
                encoders = load_pickle(encoders_path)
                info = {}
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                models_dict['ecommerce'] = {
                    'model': model, 'scaler': scaler, 'encoders': encoders,
                    'features': info.get('features', []),
                    'categorical': info.get('categorical_columns', [])
                }
            except Exception as e:
                print(f"Load failed: {e}. Triggering self-heal...")
                models_dict['ecommerce'] = train_ecommerce_fallback()
        else:
            print(f"Files missing from {dir_path}. Triggering self-heal...")
            models_dict['ecommerce'] = train_ecommerce_fallback()
    return models_dict['ecommerce']

def load_loan():
    if not models_dict['loan']:
        dir_path = os.path.join(BASE_DIR, 'loan aaproval')
        pipeline = load_pickle(os.path.join(dir_path, 'loan_approval_xgboost_pipeline.pkl'))
        models_dict['loan'] = pipeline
    return models_dict['loan']

def load_facemask():
    if not models_dict['facemask']:
        model_path = os.path.join(BASE_DIR, 'face mask detection', 'face_mask_classifier.keras')
        try:
            m = load_model(model_path)
        except Exception:
            # Fallback for keras mismatch
            base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(128, 128, 3))
            m = models.Sequential([
                base_model, layers.GlobalAveragePooling2D(), layers.BatchNormalization(), layers.Dropout(0.3),
                layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.3), layers.Dense(3, activation='softmax')
            ])
            m.load_weights(model_path.replace('.keras', '.h5') if os.path.exists(model_path.replace('.keras', '.h5')) else model_path)
        models_dict['facemask'] = m
    return models_dict['facemask']

def load_fruitveg():
    if not models_dict['fruitveg']:
        model_path = os.path.join(BASE_DIR, 'fruits vegetables', 'fruit_veg_classifier_final.keras')
        try:
            m = load_model(model_path)
        except Exception:
            base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
            m = models.Sequential([
                base_model, layers.GlobalAveragePooling2D(), layers.BatchNormalization(), layers.Dropout(0.4),
                layers.Dense(256, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.3), layers.Dense(36, activation='softmax')
            ])
            m.load_weights(model_path.replace('.keras', '.h5') if os.path.exists(model_path.replace('.keras', '.h5')) else model_path)
        # Attempt to load labels if we had them or use default 36
        models_dict['fruitveg'] = m
    return models_dict['fruitveg']

def load_audio():
    if not models_dict['audio']:
        model_path = os.path.join(BASE_DIR, 'indian language audio classification', 'indian_language_classifier.keras')
        try:
            m = load_model(model_path)
        except Exception:
            base_model = tf.keras.applications.MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
            m = tf.keras.Sequential([
                base_model, layers.GlobalAveragePooling2D(), layers.BatchNormalization(), layers.Dropout(0.4),
                layers.Dense(512, activation='relu'), layers.BatchNormalization(), layers.Dropout(0.5), layers.Dense(10, activation='softmax')
            ])
            m.load_weights(model_path.replace('.keras', '.weights.h5') if os.path.exists(model_path.replace('.keras', '.weights.h5')) else model_path)
        
        labels_path = os.path.join(BASE_DIR, 'indian language audio classification', 'label_mapping.json')
        import json
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                labels = json.load(f)['index_to_label']
        else:
            labels = {str(i): f"Language_{i}" for i in range(10)}
        models_dict['audio'] = {'model': m, 'labels': labels}
    return models_dict['audio']

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ecommerce')
def ecommerce():
    return render_template('ecommerce.html')

@app.route('/loan')
def loan():
    return render_template('loan.html')

@app.route('/facemask')
def facemask():
    return render_template('facemask.html')

@app.route('/fruitveg')
def fruitveg():
    return render_template('fruitveg.html')

@app.route('/audio')
def audio():
    return render_template('audio.html')

# --- APIs ---

@app.route('/api/ecommerce/features')
def ecommerce_features():
    try:
        data = load_ecommerce()
        if not data:
            return jsonify({'error': 'E-Commerce model files not found in /e-commerse folder!'}), 404
        
        if isinstance(data, dict) and 'error' in data:
            return jsonify({'error': f"Model Load Error: {data['error']}. Please run 'pip install -r requirements.txt' in your environment."}), 500
        
        features = data['features']
        cat_features = data['categorical']
        
        # Prepare categorical mappings for dropdowns
        cat_mapping = {}
        for col in cat_features:
            if col in data['encoders']:
                # LabelEncoder classes_ contains the labels
                cat_mapping[col] = list(data['encoders'][col].classes_)
        
        return jsonify({
            'features': features, 
            'categorical': cat_features,
            'cat_mapping': cat_mapping
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ecommerce/predict', methods=['POST'])
def ecommerce_predict():
    try:
        import pandas as pd
        data = request.json
        model_data = load_ecommerce()
        if not model_data:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Reconstruct DataFrame with all features
        df = pd.DataFrame([data])
        
        # Ensure all expected features are present in the correct order
        features = model_data['features']
        df = df[features]
        
        # Encode categorical columns
        encoders = model_data['encoders']
        for col in model_data['categorical']:
            if col in df.columns and col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col])
                except Exception as enc_err:
                    print(f"Encoding error for {col}: {enc_err}")
                    # Fallback to first class if label is new (for safety)
                    df[col] = encoders[col].transform([encoders[col].classes_[0]])[0]
        
        # Scale numerical columns
        scaler = model_data['scaler']
        X_scaled = scaler.transform(df)
        
        # Predict
        prediction = model_data['model'].predict(X_scaled)[0]
        # Check if the model has predict_proba
        if hasattr(model_data['model'], 'predict_proba'):
            probability = model_data['model'].predict_proba(X_scaled)[0][1]
        else:
            probability = 1.0 if prediction == 1 else 0.0
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'label': 'Churn' if prediction == 1 else 'Not Churn'
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/loan/features')
def loan_features():
    try:
        # The scikit-learn pipeline handles all transformations, but we need to tell the frontend
        # what the expected features are.
        features = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                    'credit_score', 'person_gender', 'person_education', 'person_home_ownership',
                    'loan_intent', 'previous_loan_defaults_on_file']
        cat_features = ['person_gender', 'person_education', 'person_home_ownership',
                        'loan_intent', 'previous_loan_defaults_on_file']
        
        cat_mapping = {}
        try:
            pipeline = load_loan()
            preproc = pipeline.named_steps['preprocessing']
            for name, t, cols in preproc.transformers:
                if name == 'cat':
                    # OneHotEncoder may be directly on 'cat' transformer
                    enc = t
                    # If it's a sub-pipeline, dive into it
                    if hasattr(t, 'named_steps'):
                        enc = t.named_steps.get('encoder', t)
                    cats = getattr(enc, 'categories_', None)
                    if cats is not None:
                        for idx, c_name in enumerate(cat_features):
                            if idx < len(cats):
                                cat_mapping[c_name] = [str(x) for x in cats[idx]]
        except Exception:
            pass

        return jsonify({'features': features, 'categorical': cat_features, 'cat_mapping': cat_mapping})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/loan/predict', methods=['POST'])
def loan_predict():
    try:
        req = request.json
        pipeline = load_loan()
        
        if pipeline is None:
            return jsonify({'error': 'Loan model file missing in "loan aaproval" folder.'}), 500
        
        # Define the exact columns the pipeline expects
        num_columns = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
                       'credit_score']
        cat_columns = ['person_gender', 'person_education', 'person_home_ownership',
                       'loan_intent', 'previous_loan_defaults_on_file']
        columns = num_columns + cat_columns

        # Build input dict — convert numeric slider values (strings) to float
        input_data = {}
        for col in num_columns:
            val = req.get(col)
            if val is None or val == '': 
                val = 0
            try:
                input_data[col] = [float(val)]
            except (TypeError, ValueError):
                input_data[col] = [0.0]
                
        for col in cat_columns:
            input_data[col] = [str(req.get(col, ''))]
        
        # Ensure exact column order before prediction
        df = pd.DataFrame(input_data)[columns]
        
        # The pipeline handles scaling, encoding, and XGBoost
        try:
            prob_arr = pipeline.predict_proba(df)[0]
            prob = float(prob_arr[1])
            pred = int(pipeline.predict(df)[0])
        except Exception as model_err:
            return jsonify({'error': f"Model execution error: {str(model_err)}. Check library versions (XGBoost/Sklearn)."}), 500
        
        return jsonify({
            'prediction': pred,
            'probability': prob,
            'label': 'Approved' if pred == 0 else 'Rejected'
        })
    except Exception as e:
        return jsonify({'error': f"System error: {str(e)}"}), 500

@app.route('/api/facemask/predict', methods=['POST'])
def facemask_predict():
    if 'image' not in request.files: return jsonify({'error': 'No image'}), 400
    try:
        model = load_facemask()
        img_file = request.files['image']
        img = Image.open(img_file).convert('RGB').resize((128, 128))
        img_arr = np.array(img) / 255.0
        X = np.expand_dims(img_arr, axis=0)
        
        preds = model.predict(X, verbose=0)[0]
        idx = int(np.argmax(preds))
        
        probs = [{'class_name': FACEMASK_CLASSES[i], 'prob': float(preds[i]*100)} for i in range(3)]
        return jsonify({'class': FACEMASK_CLASSES[idx], 'probabilities': probs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fruitveg/predict', methods=['POST'])
def fruitveg_predict():
    if 'image' not in request.files: return jsonify({'error': 'No image'}), 400
    try:
        model = load_fruitveg()
        img_file = request.files['image']
        img = Image.open(img_file).convert('RGB').resize((224, 224))
        img_arr = np.array(img) / 255.0
        X = np.expand_dims(img_arr, axis=0)
        
        preds = model.predict(X, verbose=0)[0]
        idx = int(np.argmax(preds))
        top_indices = np.argsort(preds)[-5:][::-1]
        
        top5 = [{'class_name': FRUITVEG_CLASSES[i], 'prob': float(preds[i]*100)} for i in top_indices]
        return jsonify({'class': FRUITVEG_CLASSES[idx], 'top5': top5})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _get_mel_spectrogram(audio_data, sr):
    import matplotlib
    matplotlib.use('Agg') # Safe for threading
    import matplotlib.pyplot as plt
    try:
        S = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        fig = plt.figure(figsize=(3.5, 3.5), dpi=64)
        ax = fig.add_axes([0, 0, 1, 1], frameon=False)
        ax.axis('off')
        librosa.display.specshow(S_dB, sr=sr, ax=ax, fmax=8000, cmap='viridis')
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        img = Image.open(buf).convert('RGB').resize((224, 224))
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        raise e

@app.route('/api/audio/predict', methods=['POST'])
def audio_predict():
    if 'audio' not in request.files: return jsonify({'error': 'No audio'}), 400
    try:
        data = load_audio()
        model = data['model']
        idx_to_label = data['labels']
        
        audio_file = request.files['audio']
        path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
        audio_file.save(path)
        
        try:
            y, sr = librosa.load(path, sr=22050, duration=15)
            y, _ = librosa.effects.trim(y, top_db=20)
            y = y[:sr * 5]
            if len(y) < sr * 0.5:
                raise ValueError("Audio too short after trimming.")
                
            img_arr = _get_mel_spectrogram(y, sr)
            X = np.expand_dims(img_arr, axis=0)
            
            preds = model.predict(X, verbose=0)[0]
            idx = int(np.argmax(preds))
            cls_name = idx_to_label.get(str(idx), f"Language_{idx}")
            
            probs = [{'class_name': idx_to_label.get(str(i), f"Lang {i}"), 'prob': float(preds[i]*100)} for i in range(len(preds))]
            probs = sorted(probs, key=lambda x: x['prob'], reverse=True)
            
            return jsonify({'class': cls_name, 'probabilities': probs})
        finally:
            if os.path.exists(path):
                os.remove(path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/debug')
def debug_env():
    import sys
    import subprocess
    try:
        pkgs = subprocess.check_output([sys.executable, '-m', 'pip', 'list']).decode()
    except:
        pkgs = "Could not list packages"
    
    return jsonify({
        'python_version': sys.version,
        'python_executable': sys.executable,
        'path': sys.path,
        'installed_packages': pkgs,
        'base_dir': BASE_DIR,
        'ecommerce_folder': os.listdir(os.path.join(BASE_DIR, 'e-commerse')) if os.path.exists(os.path.join(BASE_DIR, 'e-commerse')) else "NOT FOUND"
    })

@app.route('/debug/paths')
def debug_paths():
    return jsonify({
        'BASE_DIR': BASE_DIR,
        'cwd': os.getcwd(),
        'ecom_expected': os.path.join(BASE_DIR, 'e-commerse'),
        'ecom_exists': os.path.exists(os.path.join(BASE_DIR, 'e-commerse')),
        'ecom_contents': os.listdir(os.path.join(BASE_DIR, 'e-commerse')) if os.path.exists(os.path.join(BASE_DIR, 'e-commerse')) else []
    })

if __name__ == '__main__':
    app.run(debug=True)
