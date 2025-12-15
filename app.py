"""
Halal Food Scanner - Flask Backend
FYP: Japanese Snack Label Classification
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import io
import re
import cv2
import numpy as np
import pandas as pd
import pickle
import easyocr
from PIL import Image
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
import torch

# ------------------------------------------------------------------------------
# FIX: Monkey Patch for 'ANTIALIAS' error
# ------------------------------------------------------------------------------
# Many libraries (like older EasyOCR versions) still try to access 
# Image.ANTIALIAS which was removed in Pillow 10.0.0. 
# We manually restore it here to prevent crashes in dependencies.
if not hasattr(Image, 'ANTIALIAS'):
    try:
        Image.ANTIALIAS = Image.Resampling.LANCZOS
    except AttributeError:
        pass

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models (loaded once at startup)
reader = None
sentence_model = None
classifier_model = None
label_encoder = None
ingredient_dict = None

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def translate_text(text, source='ja', target='en'):
    """Translates text using GoogleTranslator."""
    try:
        if not text or text.isspace():
            return ""
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        print(f"Translation failed: {e}")
        return "[Translation failed]"

def clean_text_for_matching(text):
    """Cleans text for rule-based matching."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------------------------------------------------------
# MODEL LOADING (Run once at startup)
# ------------------------------------------------------------------------------

def load_models():
    """Load all models at startup."""
    global reader, sentence_model, classifier_model, label_encoder, ingredient_dict
    
    print("🔧 Loading EasyOCR...")
    reader = easyocr.Reader(['ja', 'en'], gpu=torch.cuda.is_available())
    print(f"✅ EasyOCR loaded (GPU: {torch.cuda.is_available()})")
    
    print("🧠 Loading Sentence Transformer...")
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ Sentence Transformer loaded")
    
    print("📖 Loading ingredient dictionary...")
    try:
        item_df = pd.read_excel('dataset.xlsx')
        ingredient_dict = dict(zip(
            item_df['ingredients'].str.lower().str.strip(), 
            item_df['labels'].str.upper().str.strip()
        ))
        print(f"✅ Loaded {len(ingredient_dict)} ingredients")
    except Exception as e:
        print(f"❌ Error loading dataset.xlsx: {e}")
        ingredient_dict = {}
    
    print("🤖 Loading trained classifier model...")
    try:
        with open(os.path.join(MODEL_FOLDER, 'classifier_model.pkl'), 'rb') as f:
            classifier_model = pickle.load(f)
        with open(os.path.join(MODEL_FOLDER, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Classifier model loaded")
    except Exception as e:
        print(f"⚠️ Model not found. Please train and save the model first: {e}")
        classifier_model = None
        label_encoder = None

# ------------------------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------------------------

def predict_snack(image_bytes):
    """
    Full pipeline: OCR → Translation → Rule-based + ML Classification
    Returns dictionary with results
    """
    results = {
        "japanese_text": "",
        "translated_text": "",
        "matched_ingredients": [],
        "rule_based_status": "UNDOUBTFUL",
        "model_prediction": "UNDOUBTFUL",
        "final_decision": "UNDOUBTFUL",
        "confidence": 0.0,
        "error": None
    }
    
    try:
        # 1. Open image with PIL
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # ensure 3 channels

        # Resize using modern LANCZOS filter
        # Note: The global patch at the top ensures compatibility if libraries use ANTIALIAS
        if hasattr(Image, 'Resampling'):
            img = img.resize((800, 800), Image.Resampling.LANCZOS)
        else:
            img = img.resize((800, 800), Image.ANTIALIAS)

        # Convert to OpenCV format for EasyOCR
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # 2. OCR
        ocr_results = reader.readtext(img_array)
        japanese_text = ' '.join([text for (bbox, text, conf) in ocr_results])
        
        if not japanese_text:
            results["error"] = "OCR could not extract any text from the image."
            return results
        results["japanese_text"] = japanese_text

        # 3. Translation
        translated_text = translate_text(japanese_text)
        if not translated_text or translated_text == "[Translation failed]":
            results["error"] = "Translation failed. Please try again."
            return results
        results["translated_text"] = translated_text

        # 4. Rule-Based Matching
        cleaned_text = clean_text_for_matching(translated_text)
        found_haram = False
        found_doubtful = False
        
        for ingredient, status in ingredient_dict.items():
            pattern = r'\b' + re.escape(ingredient) + r'\b'
            
            if re.search(pattern, cleaned_text):
                results["matched_ingredients"].append({
                    "ingredient": ingredient.title(),
                    "status": status
                })
                
                if status == 'HARAM':
                    found_haram = True
                elif status == 'DOUBTFUL':
                    found_doubtful = True
        
        if found_haram:
            results["rule_based_status"] = 'HARAM'
        elif found_doubtful:
            results["rule_based_status"] = 'DOUBTFUL'
        else:
            results["rule_based_status"] = 'UNDOUBTFUL'
        
        # 5. ML Model Prediction
        if classifier_model and sentence_model:
            text_embedding = sentence_model.encode([translated_text])
            prediction_encoded = classifier_model.predict(text_embedding)[0]
            
            if hasattr(classifier_model, 'predict_proba'):
                proba = classifier_model.predict_proba(text_embedding)[0]
                results["confidence"] = float(max(proba) * 100)
            
            prediction_label = label_encoder.inverse_transform([prediction_encoded])[0].upper()
            results["model_prediction"] = prediction_label
        else:
            results["model_prediction"] = results["rule_based_status"]
            results["confidence"] = 0.0
        
        # 6. Final Decision (Haram > Doubtful > Undoubtful)
        if results["rule_based_status"] == 'HARAM' or results["model_prediction"] == 'HARAM':
            results["final_decision"] = 'HARAM'
        elif results["rule_based_status"] == 'DOUBTFUL' or results["model_prediction"] == 'DOUBTFUL':
            results["final_decision"] = 'DOUBTFUL'
        else:
            results["final_decision"] = 'UNDOUBTFUL'

        return results

    except Exception as e:
        results["error"] = f"An error occurred: {str(e)}"
        return results


# ------------------------------------------------------------------------------
# FLASK ROUTES
# ------------------------------------------------------------------------------

@app.route('/')
def index():
    """Serve the main PWA page."""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_image():
    """API endpoint for image analysis."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file:
        try:
            # Read image bytes
            image_bytes = file.read()
            
            # Run prediction
            results = predict_snack(image_bytes)
            
            if results['error']:
                return jsonify({'error': results['error']}), 400
            
            return jsonify(results), 200
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid request'}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'ocr_loaded': reader is not None,
        'model_loaded': classifier_model is not None,
        'ingredients_loaded': len(ingredient_dict) if ingredient_dict else 0
    }), 200

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Halal Food Scanner API")
    print("=" * 60)
    
    # Load models at startup
    load_models()
    
    print("\nServer ready:")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)