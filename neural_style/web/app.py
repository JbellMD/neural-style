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


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_style_transfer_model(method='vgg', device=None):
    """Get or initialize style transfer model."""
    global style_transfer_model
    
    if style_transfer_model is None or style_transfer_model.method != method:
        style_transfer_model = StyleTransfer(method=method, device=device)
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


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', styles=get_available_styles())


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
    return send_from_directory(static_dir, path)


@app.route('/api/styles')
def get_styles():
    """Get available styles."""
    return jsonify(get_available_styles())


@app.route('/api/transfer', methods=['POST'])
def transfer_style():
    """Apply style transfer to an image."""
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
        
        result = model.transfer(
            content_image=content_path,
            style_image=style_path,
            output_path=output_path,
            image_size=image_size,
            style_weight=style_weight,
            content_weight=content_weight,
            show_progress=False
        )
        
        elapsed_time = time.time() - start_time
        
        # Return result
        return jsonify({
            'success': True,
            'resultUrl': f'/static/results/{output_filename}',
            'processingTime': round(elapsed_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Error during style transfer: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/transfer-base64', methods=['POST'])
def transfer_style_base64():
    """Apply style transfer to base64-encoded images."""
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
        
        # Decode base64 content image
        content_data = base64.b64decode(data['contentImage'].split(',')[1])
        content_image = Image.open(io.BytesIO(content_data))
        
        # Save content image
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], f"content_{uuid.uuid4()}.jpg")
        content_image.save(content_path)
        
        # Get parameters
        method = data.get('method', 'vgg')
        image_size = int(data.get('imageSize', 512))
        style_weight = float(data.get('styleWeight', 1e6))
        content_weight = float(data.get('contentWeight', 1.0))
        
        # Generate output filename
        output_filename = f"result_{uuid.uuid4()}.jpg"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        
        # Apply style transfer
        model = get_style_transfer_model(method=method)
        start_time = time.time()
        
        result = model.transfer(
            content_image=content_path,
            style_image=style_path,
            output_path=output_path,
            image_size=image_size,
            style_weight=style_weight,
            content_weight=content_weight,
            show_progress=False
        )
        
        elapsed_time = time.time() - start_time
        
        # Convert result to base64
        with open(output_path, 'rb') as f:
            result_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Return result
        return jsonify({
            'success': True,
            'resultUrl': f'/static/results/{output_filename}',
            'resultImage': f'data:image/jpeg;base64,{result_base64}',
            'processingTime': round(elapsed_time, 2)
        })
        
    except Exception as e:
        logger.error(f"Error during style transfer: {str(e)}")
        return jsonify({'error': str(e)}), 500


def run_app(host='0.0.0.0', port=8000, debug=False):
    """Run the Flask app."""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_app(debug=True)
