"""
Web interface for neural style transfer.
"""

import os
import io
import uuid
import time
import json
import base64
from pathlib import Path
from PIL import Image

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from ..style_transfer import StyleTransfer
from ..utils.image_utils import load_image, save_image
from ..utils.logging import get_logger

# Setup logger
logger = get_logger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')
STYLES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'styles')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(STYLES_FOLDER, exist_ok=True)

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['STYLES_FOLDER'] = STYLES_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Initialize style transfer model
style_transfer_model = None

# Available style transfer methods
METHODS = {
    'vgg': {
        'id': 'vgg',
        'name': 'VGG (Gatys)',
        'description': 'Original neural style transfer algorithm. Produces high-quality results but slower.',
        'params': ['iterations', 'styleWeight', 'contentWeight']
    },
    'fast': {
        'id': 'fast',
        'name': 'Fast Neural Style',
        'description': 'Real-time style transfer using a pre-trained model. Very fast but limited to trained styles.',
        'params': []
    },
    'adain': {
        'id': 'adain',
        'name': 'AdaIN',
        'description': 'Adaptive Instance Normalization for fast arbitrary style transfer with good quality.',
        'params': ['alpha']
    },
    'attention': {
        'id': 'attention',
        'name': 'Attention-based',
        'description': 'Uses self-attention mechanisms to better capture and transfer style patterns.',
        'params': ['alpha']
    }
}

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_style_transfer_model(method='vgg', device=None):
    """Get or initialize style transfer model."""
    global style_transfer_model
    
    if style_transfer_model is None or style_transfer_model.method != method:
        # For fast method, try to find a pre-trained model
        model_path = None
        if method == 'fast':
            model_dir = os.path.join('models', 'fast')
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                if model_files:
                    model_path = os.path.join(model_dir, model_files[0])
                    logger.info(f"Using pre-trained fast model: {model_path}")
        
        style_transfer_model = StyleTransfer(method=method, device=device, model_path=model_path)
        logger.info(f"Initialized style transfer model with method: {method}")
    
    return style_transfer_model


def get_available_styles():
    """Get list of available pre-defined styles."""
    styles_dir = Path(app.config['STYLES_FOLDER'])
    style_files = list(styles_dir.glob('*.jpg')) + list(styles_dir.glob('*.jpeg')) + list(styles_dir.glob('*.png'))
    
    styles = []
    for style_file in style_files:
        styles.append({
            'id': style_file.stem,
            'name': style_file.stem.replace('_', ' ').title(),
            'path': f'/static/styles/{style_file.name}'
        })
    
    return styles


def get_available_methods():
    """Get list of available style transfer methods with descriptions."""
    return list(METHODS.values())


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', 
                          styles=get_available_styles(),
                          methods=get_available_methods())


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, path)


@app.route('/api/styles')
def get_styles():
    """Get available styles."""
    return jsonify(get_available_styles())


@app.route('/api/methods')
def get_methods():
    """Get available style transfer methods."""
    return jsonify(get_available_methods())


@app.route('/api/transfer', methods=['POST'])
def transfer_style():
    try:
        # Check if content and style files are provided
        if 'content' not in request.files:
            return jsonify({'error': 'No content image provided'}), 400
        
        content_file = request.files['content']
        if content_file.filename == '':
            return jsonify({'error': 'No content image selected'}), 400
        
        if not allowed_file(content_file.filename):
            return jsonify({'error': 'Invalid content image format'}), 400
        
        # Get style image (either from file upload or predefined style)
        style_path = None
        if 'style' in request.files and request.files['style'].filename != '':
            style_file = request.files['style']
            if not allowed_file(style_file.filename):
                return jsonify({'error': 'Invalid style image format'}), 400
            
            # Save style image
            style_filename = secure_filename(style_file.filename)
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], f"style_{uuid.uuid4()}_{style_filename}")
            style_file.save(style_path)
        elif 'styleId' in request.form:
            style_id = request.form['styleId']
            style_path = os.path.join(app.config['STYLES_FOLDER'], f"{style_id}.jpg")
            if not os.path.exists(style_path):
                return jsonify({'error': f'Style not found: {style_id}'}), 404
        else:
            return jsonify({'error': 'No style image or style ID provided'}), 400
        
        # Get parameters
        method = request.form.get('method', 'vgg')
        image_size = int(request.form.get('imageSize', 512))
        style_weight = float(request.form.get('styleWeight', 1e6))
        content_weight = float(request.form.get('contentWeight', 1.0))
        iterations = int(request.form.get('iterations', 300))
        alpha = float(request.form.get('alpha', 1.0))
        
        # Check if method is valid
        if method not in METHODS:
            return jsonify({'error': f'Invalid method: {method}'}), 400
        
        # Save content image
        content_filename = secure_filename(content_file.filename)
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], f"content_{uuid.uuid4()}_{content_filename}")
        content_file.save(content_path)
        
        # Generate output filename
        output_filename = f"result_{uuid.uuid4()}.jpg"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        # Apply style transfer
        model = get_style_transfer_model(method=method)
        start_time = time.time()
        
        # Set parameters based on method
        params = {
            'output_image': output_path,
            'content_scale': image_size / 512.0,  # Scale relative to 512px
        }
        
        if method == 'vgg':
            params.update({
                'content_weight': content_weight,
                'style_weight': style_weight,
                'iterations': iterations
            })
        elif method in ['adain', 'attention']:
            params.update({
                'alpha': alpha
            })
        
        # Apply style transfer with the selected method
        model.transfer(
            content_path, 
            style_path, 
            method=method,
            **params
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        # Log the result
        logger.info(f"Style transfer completed: {method} method, {processing_time}s, size={image_size}")
        
        # Return result
        result_url = f'/static/results/{output_filename}'
        return jsonify({
            'success': True,
            'resultUrl': result_url,
            'method': method,
            'processingTime': processing_time
        })
        
    except Exception as e:
        logger.error(f"Error during style transfer: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/transfer-base64', methods=['POST'])
def transfer_style_base64():
    try:
        data = request.get_json()
        
        # Check if content and style images are provided
        if 'contentImage' not in data:
            return jsonify({'error': 'No content image provided'}), 400
        
        # Get style image (either from base64 or predefined style)
        style_path = None
        if 'styleImage' in data and data['styleImage']:
            # Decode base64 style image
            style_data = base64.b64decode(data['styleImage'].split(',')[1])
            style_image = Image.open(io.BytesIO(style_data))
            
            # Save style image
            style_path = os.path.join(app.config['UPLOAD_FOLDER'], f"style_{uuid.uuid4()}.jpg")
            style_image.save(style_path)
        elif 'styleId' in data:
            style_id = data['styleId']
            style_path = os.path.join(app.config['STYLES_FOLDER'], f"{style_id}.jpg")
            if not os.path.exists(style_path):
                return jsonify({'error': f'Style not found: {style_id}'}), 404
        else:
            return jsonify({'error': 'No style image or style ID provided'}), 400
        
        # Get parameters
        method = data.get('method', 'vgg')
        image_size = int(data.get('imageSize', 512))
        style_weight = float(data.get('styleWeight', 1e6))
        content_weight = float(data.get('contentWeight', 1.0))
        iterations = int(data.get('iterations', 300))
        alpha = float(data.get('alpha', 1.0))
        
        # Check if method is valid
        if method not in METHODS:
            return jsonify({'error': f'Invalid method: {method}'}), 400
        
        # Decode base64 content image
        content_data = base64.b64decode(data['contentImage'].split(',')[1])
        content_image = Image.open(io.BytesIO(content_data))
        
        # Save content image
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], f"content_{uuid.uuid4()}.jpg")
        content_image.save(content_path)
        
        # Generate output filename
        output_filename = f"result_{uuid.uuid4()}.jpg"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        # Apply style transfer
        model = get_style_transfer_model(method=method)
        start_time = time.time()
        
        # Set parameters based on method
        params = {
            'output_image': output_path,
            'content_scale': image_size / 512.0,  # Scale relative to 512px
        }
        
        if method == 'vgg':
            params.update({
                'content_weight': content_weight,
                'style_weight': style_weight,
                'iterations': iterations
            })
        elif method in ['adain', 'attention']:
            params.update({
                'alpha': alpha
            })
        
        # Apply style transfer with the selected method
        model.transfer(
            content_path, 
            style_path, 
            method=method,
            **params
        )
        
        # Convert result to base64
        buffer = io.BytesIO()
        result = Image.open(output_path)
        result.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        processing_time = round(time.time() - start_time, 2)
        
        # Log the result
        logger.info(f"Style transfer completed: {method} method, {processing_time}s, size={image_size}")
        
        # Return result
        return jsonify({
            'success': True,
            'resultUrl': f'/static/results/{output_filename}',
            'resultImage': f'data:image/jpeg;base64,{img_str}',
            'processingTime': processing_time,
            'method': method
        })
        
    except Exception as e:
        logger.error(f"Error during style transfer: {str(e)}")
        return jsonify({'error': str(e)}), 500


def run_app(host='0.0.0.0', port=8000, debug=False):
    """Run the Flask app."""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app(debug=True)
