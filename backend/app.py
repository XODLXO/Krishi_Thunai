# backend/app.py
import os
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from backend.core.predictor import predict_image
from backend.core.severity import estimate_severity

app = Flask(__name__, static_folder='../frontend', static_url_path='/')

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception as e:
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400

        try:
            prediction_result = predict_image(img)
            severity_result = estimate_severity(img)
            
            response_data = {
                "success": True,
                "prediction": prediction_result["class"],
                "confidence": prediction_result["confidence"],
                "severity_score": severity_result["severity_score"]
            }
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Inference Error: {e}")
            return jsonify({'error': f'An error occurred during model inference: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)