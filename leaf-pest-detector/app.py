"""
🌿 Leaf Pest Detection — Flask Application (PyTorch)
=====================================================
Upload a leaf image → get pest/healthy prediction.

Usage:
  python app.py

Then open http://localhost:5000
"""

import os
import json
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify, render_template, send_from_directory

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'leaf_pest_model.pth')
CONFIG_PATH = os.path.join(MODEL_DIR, 'model_config.json')
SAMPLE_DIR = os.path.join(BASE_DIR, 'sample_images')
IMG_SIZE = 224

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default class info (used in demo mode)
DEFAULT_CONFIG = {
    'class_names': ['aphid', 'caterpillar', 'healthy'],
    'is_demo': True,
    'num_classes': 3,
    'descriptions': {
        'healthy': {
            'name': 'Healthy Leaf',
            'emoji': '✅',
            'description': 'The leaf appears healthy with no signs of pest damage. Good plant health indicators observed.',
            'recommendation': 'Continue regular care and monitoring. No treatment needed.'
        },
        'aphid': {
            'name': 'Aphid Infestation',
            'emoji': '🐛',
            'description': 'Aphids detected on the leaf surface. These small insects suck plant sap, causing leaf curling and yellowing.',
            'recommendation': 'Apply neem oil spray or introduce ladybugs as natural predators. Remove heavily infested leaves.'
        },
        'caterpillar': {
            'name': 'Caterpillar Damage',
            'emoji': '🦋',
            'description': 'Signs of caterpillar feeding detected. Irregular holes and chewed leaf edges indicate active caterpillar presence.',
            'recommendation': 'Hand-pick caterpillars if few. Apply Bt (Bacillus thuringiensis) spray for larger infestations.'
        }
    }
}


# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)

# Global state
model = None
config = DEFAULT_CONFIG

# Image transform for inference
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def build_model(num_classes=3):
    """Build MobileNetV2 model architecture."""
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(m.last_channel, num_classes)
    )
    return m


def load_model_and_config():
    """Load model and config, fall back to demo mode if not available."""
    global model, config

    # Load config
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        print(f"📋 Loaded config: {len(config['class_names'])} classes")
    else:
        config = DEFAULT_CONFIG
        print("📋 Using default config (demo mode)")

    # Load model
    if os.path.exists(MODEL_PATH):
        try:
            num_classes = config.get('num_classes', 3)
            model = build_model(num_classes)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
            model.eval()
            print(f"🧠 Model loaded from {MODEL_PATH}")
            if config.get('is_demo'):
                print("   ⚠️  Demo model — using color analysis for predictions")
            else:
                print("   ✅ Trained model ready")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            print("   Falling back to demo mode")
            model = None
    else:
        print("⚠️  No model found — running in demo mode")
        print(f"   Create one with:  python train.py --demo")


def analyze_leaf_image(img):
    """
    Analyze leaf image using color heuristics for demo mode.
    Returns plausible class probabilities based on image characteristics.
    """
    img_array = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

    # Analyze color channels
    r_mean = np.mean(img_array[:, :, 0])
    g_mean = np.mean(img_array[:, :, 1])
    b_mean = np.mean(img_array[:, :, 2])

    # Texture analysis
    r_std = np.std(img_array[:, :, 0])
    g_std = np.std(img_array[:, :, 1])

    # Green dominance
    green_ratio = g_mean / (r_mean + g_mean + b_mean + 1e-6)

    # Generate plausible probabilities [aphid, caterpillar, healthy]
    if green_ratio > 0.38:
        probs = np.array([0.08, 0.05, 0.87])
    elif green_ratio < 0.30:
        if r_mean > g_mean:
            probs = np.array([0.72, 0.18, 0.10])
        else:
            probs = np.array([0.15, 0.73, 0.12])
    else:
        if r_std > 0.15:
            probs = np.array([0.55, 0.30, 0.15])
        elif g_std > 0.13:
            probs = np.array([0.20, 0.60, 0.20])
        else:
            probs = np.array([0.12, 0.08, 0.80])

    # Add small random variation
    noise = np.random.uniform(-0.03, 0.03, 3)
    probs = np.clip(probs + noise, 0.01, 0.99)
    probs = probs / probs.sum()

    return probs


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main UI page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict pest/healthy from uploaded leaf image."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Open image
        img = Image.open(file).convert('RGB')

        # Determine if using real model or demo
        is_demo = config.get('is_demo', True)

        if model is not None and not is_demo:
            # ── Real Model Prediction ──
            input_tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
        else:
            # ── Demo Mode: Color Analysis ──
            probs = analyze_leaf_image(img)

        # Get results
        class_names = config['class_names']
        pred_idx = int(np.argmax(probs))
        pred_class = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        # Get description info
        descriptions = config.get('descriptions', DEFAULT_CONFIG['descriptions'])
        class_info = descriptions.get(pred_class, {
            'name': pred_class.title(),
            'emoji': '🔍',
            'description': f'Detected: {pred_class}',
            'recommendation': 'Consult an agricultural expert.'
        })

        # All class probabilities
        all_probs = {
            class_names[i]: {
                'probability': float(probs[i]),
                'info': descriptions.get(class_names[i], {}).get('name', class_names[i].title())
            }
            for i in range(len(class_names))
        }

        return jsonify({
            'success': True,
            'prediction': {
                'class': pred_class,
                'name': class_info['name'],
                'emoji': class_info['emoji'],
                'confidence': confidence,
                'description': class_info['description'],
                'recommendation': class_info['recommendation'],
                'is_healthy': pred_class == 'healthy'
            },
            'all_probabilities': all_probs,
            'is_demo_mode': is_demo
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/sample_images/<path:filename>')
def serve_sample_image(filename):
    """Serve sample images for demo."""
    return send_from_directory(SAMPLE_DIR, filename)


@app.route('/api/samples')
def list_samples():
    """List available sample images."""
    samples = []
    if os.path.exists(SAMPLE_DIR):
        for f in sorted(os.listdir(SAMPLE_DIR)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                name = os.path.splitext(f)[0].replace('_', ' ').title()
                samples.append({
                    'filename': f,
                    'name': name,
                    'url': f'/sample_images/{f}'
                })
    return jsonify({'samples': samples})


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  🌿 Leaf Pest Detection — Starting Server")
    print("=" * 55)
    print()

    load_model_and_config()

    print()
    print("🌐 Open in browser:  http://localhost:5000")
    print("=" * 55)
    print()

    app.run(debug=True, host='0.0.0.0', port=5000)
