from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Clear any existing registry to avoid duplicate metrics
# (This helps if you're reloading the app multiple times)
for collector in list(REGISTRY._collector_to_names.keys()):
    REGISTRY.unregister(collector)

# Prometheus metrics - FIXED: 'class' is a reserved keyword, use 'pred_class' instead
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total number of predictions', ['pred_class'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency in seconds')
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status_code'])

# Load model at startup (with error handling)
try:
    logger.info("Loading MobileNetV2 model...")
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# HTML template for web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Image Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .result { margin-top: 20px; padding: 20px; border: 1px solid #ccc; border-radius: 5px; }
        .prediction { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 3px; }
        .confidence { color: #0066cc; font-weight: bold; }
        .header { background: #4CAF50; color: white; padding: 20px; text-align: center; border-radius: 5px; margin-bottom: 20px; }
        .upload-form { border: 2px dashed #ccc; padding: 30px; text-align: center; border-radius: 5px; }
        button { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
        button:hover { background: #45a049; }
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Image Classifier</h1>
        <p>Upload an image to classify it using MobileNetV2</p>
    </div>
    
    <div class="upload-form">
        <form method="post" enctype="multipart/form-data" action="/upload">
            <input type="file" name="image" accept="image/*" required style="margin: 20px;">
            <br>
            <button type="submit">Classify Image</button>
        </form>
    </div>
    
    {% if predictions %}
    <div class="result">
        <h3>Classification Results:</h3>
        {% for pred in predictions %}
        <div class="prediction">
            <strong>{{ pred.class }}</strong> 
            <span class="confidence">({{ "%.2f"|format(pred.confidence*100) }}% confidence)</span>
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    <div style="margin-top: 30px; text-align: center;">
        <p><a href="/metrics">View Metrics</a> | <a href="/health">Health Check</a></p>
        <p style="color: #666; font-size: 0.8em;">Ready for Kubernetes deployment!</p>
    </div>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, predictions=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    
    if 'image' not in request.files:
        REQUEST_COUNT.labels(method='POST', endpoint='/upload', status_code='400').inc()
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        image_file = request.files['image']
        
        # Check if file was uploaded
        if image_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Read and process image
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        # Preprocess image
        image = image.resize((224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Make prediction
        with PREDICTION_LATENCY.time():
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500
            predictions = model.predict(image_array, verbose=0)
        
        # Decode predictions
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]
        results = [{'class': pred[1], 'confidence': float(pred[2])} for pred in decoded]
        
        # Update metrics - FIXED: using 'pred_class' instead of 'class'
        for pred in decoded:
            PREDICTIONS_TOTAL.labels(pred_class=pred[1]).inc()
        
        REQUEST_COUNT.labels(method='POST', endpoint='/upload', status_code='200').inc()
        
        # Log prediction latency
        latency = time.time() - start_time
        logger.info(f"Prediction completed in {latency:.2f} seconds")
        
        # For web UI
        if request.accept_mimetypes.accept_html:
            return render_template_string(HTML_TEMPLATE, predictions=results)
        
        # For API
        return jsonify({
            'predictions': results,
            'latency_seconds': latency
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        REQUEST_COUNT.labels(method='POST', endpoint='/upload', status_code='500').inc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    if model is None:
        return jsonify({
            'status': 'unhealthy', 
            'reason': 'Model not loaded',
            'timestamp': time.time()
        }), 500
    
    return jsonify({
        'status': 'healthy',
        'model': 'MobileNetV2',
        'ready': True,
        'timestamp': time.time()
    })

@app.route('/ready')
def ready():
    return jsonify({
        'ready': model is not None,
        'timestamp': time.time()
    })

@app.route('/metrics')
def metrics():
    return generate_latest()

if __name__ == '__main__':
    # For local development
    app.run(host='0.0.0.0', port=5000, debug=False)
