/**
 * Neural Style Transfer Web UI
 * Main JavaScript file for handling UI interactions and API calls
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const contentImageInput = document.getElementById('contentImageInput');
    const contentImagePreview = document.getElementById('contentImagePreview');
    const contentImage = document.getElementById('contentImage');
    const contentImageContainer = document.getElementById('contentImageContainer');
    
    const styleImageInput = document.getElementById('styleImageInput');
    const styleImagePreview = document.getElementById('styleImagePreview');
    const styleImage = document.getElementById('styleImage');
    const styleImageContainer = document.getElementById('styleImageContainer');
    
    const resultImage = document.getElementById('resultImage');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    const styleGallery = document.getElementById('styleGallery');
    
    const transferButton = document.getElementById('transferButton');
    const downloadButton = document.getElementById('downloadButton');
    
    const methodSelect = document.getElementById('methodSelect');
    const methodDescription = document.getElementById('methodDescription');
    const imageSize = document.getElementById('imageSize');
    
    // Parameter controls
    const iterationsContainer = document.getElementById('iterationsContainer');
    const iterations = document.getElementById('iterations');
    const iterationsValue = document.getElementById('iterationsValue');
    
    const styleWeightContainer = document.getElementById('styleWeightContainer');
    const styleWeight = document.getElementById('styleWeight');
    const styleWeightValue = document.getElementById('styleWeightValue');
    
    const contentWeightContainer = document.getElementById('contentWeightContainer');
    const contentWeight = document.getElementById('contentWeight');
    const contentWeightValue = document.getElementById('contentWeightValue');
    
    const alphaContainer = document.getElementById('alphaContainer');
    const alpha = document.getElementById('alpha');
    const alphaValue = document.getElementById('alphaValue');
    
    // State
    let contentImageData = null;
    let styleImageData = null;
    let selectedStyleId = null;
    let resultImageUrl = null;
    
    // Method descriptions
    const methodDescriptions = {
        'vgg': 'Original neural style transfer algorithm. Produces high-quality results but slower.',
        'fast': 'Real-time style transfer using a pre-trained model. Very fast but limited to trained styles.',
        'adain': 'Adaptive Instance Normalization for fast arbitrary style transfer with good quality.',
        'attention': 'Uses self-attention mechanisms to better capture and transfer style patterns.'
    };
    
    // Method parameters
    const methodParams = {
        'vgg': {
            iterations: true,
            styleWeight: true,
            contentWeight: true,
            alpha: false
        },
        'fast': {
            iterations: false,
            styleWeight: false,
            contentWeight: false,
            alpha: false
        },
        'adain': {
            iterations: false,
            styleWeight: false,
            contentWeight: false,
            alpha: true
        },
        'attention': {
            iterations: false,
            styleWeight: false,
            contentWeight: false,
            alpha: true
        }
    };
    
    // Initialize
    loadPredefinedStyles();
    setupEventListeners();
    updateMethodParams();
    
    /**
     * Load predefined styles from the server
     */
    function loadPredefinedStyles() {
        fetch('/api/styles')
            .then(response => response.json())
            .then(styles => {
                if (styles.length > 0) {
                    styles.forEach(style => {
                        const styleItem = document.createElement('div');
                        styleItem.className = 'style-item';
                        styleItem.dataset.styleId = style.id;
                        
                        const styleImg = document.createElement('img');
                        styleImg.src = style.path;
                        styleImg.alt = style.name;
                        styleImg.className = 'img-fluid';
                        
                        const styleName = document.createElement('div');
                        styleName.className = 'style-name';
                        styleName.textContent = style.name;
                        
                        styleItem.appendChild(styleImg);
                        styleItem.appendChild(styleName);
                        styleGallery.appendChild(styleItem);
                        
                        styleItem.addEventListener('click', () => {
                            selectPredefinedStyle(style.id, style.path);
                        });
                    });
                }
            })
            .catch(error => {
                console.error('Error loading styles:', error);
            });
    }
    
    /**
     * Set up event listeners for UI interactions
     */
    function setupEventListeners() {
        // Content image upload
        contentImageContainer.addEventListener('click', () => {
            contentImageInput.click();
        });
        
        contentImageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                const reader = new FileReader();
                
                reader.onload = (event) => {
                    contentImageData = event.target.result;
                    contentImage.src = contentImageData;
                    contentImage.classList.remove('d-none');
                    contentImagePreview.querySelector('.upload-placeholder').classList.add('d-none');
                    updateTransferButtonState();
                };
                
                reader.readAsDataURL(file);
            }
        });
        
        // Style image upload
        styleImageContainer.addEventListener('click', () => {
            styleImageInput.click();
        });
        
        styleImageInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                const reader = new FileReader();
                
                reader.onload = (event) => {
                    styleImageData = event.target.result;
                    styleImage.src = styleImageData;
                    styleImage.classList.remove('d-none');
                    styleImagePreview.querySelector('.upload-placeholder').classList.add('d-none');
                    
                    // Clear selected style
                    selectedStyleId = null;
                    document.querySelectorAll('.style-item.selected').forEach(item => {
                        item.classList.remove('selected');
                    });
                    
                    updateTransferButtonState();
                };
                
                reader.readAsDataURL(file);
            }
        });
        
        // Transfer button
        transferButton.addEventListener('click', applyStyleTransfer);
        
        // Download button
        downloadButton.addEventListener('click', () => {
            if (resultImageUrl) {
                const link = document.createElement('a');
                link.href = resultImageUrl;
                link.download = 'styled_image.jpg';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        });
        
        // Method select
        methodSelect.addEventListener('change', updateMethodParams);
        
        // Parameter sliders
        iterations.addEventListener('input', () => {
            iterationsValue.textContent = iterations.value;
        });
        
        styleWeight.addEventListener('input', () => {
            styleWeightValue.textContent = `1e${styleWeight.value}`;
        });
        
        contentWeight.addEventListener('input', () => {
            contentWeightValue.textContent = contentWeight.value;
        });
        
        alpha.addEventListener('input', () => {
            alphaValue.textContent = alpha.value;
        });
    }
    
    /**
     * Update parameters based on selected method
     */
    function updateMethodParams() {
        const method = methodSelect.value;
        
        // Update method description
        methodDescription.textContent = methodDescriptions[method];
        
        // Show/hide parameters based on method
        if (methodParams[method].iterations) {
            iterationsContainer.classList.remove('d-none');
        } else {
            iterationsContainer.classList.add('d-none');
        }
        
        if (methodParams[method].styleWeight) {
            styleWeightContainer.classList.remove('d-none');
        } else {
            styleWeightContainer.classList.add('d-none');
        }
        
        if (methodParams[method].contentWeight) {
            contentWeightContainer.classList.remove('d-none');
        } else {
            contentWeightContainer.classList.add('d-none');
        }
        
        if (methodParams[method].alpha) {
            alphaContainer.classList.remove('d-none');
        } else {
            alphaContainer.classList.add('d-none');
        }
    }
    
    /**
     * Select a predefined style
     * @param {string} styleId - ID of the selected style
     * @param {string} stylePath - Path to the style image
     */
    function selectPredefinedStyle(styleId, stylePath) {
        // Update selected style
        selectedStyleId = styleId;
        
        // Update style image preview
        styleImage.src = stylePath;
        styleImage.classList.remove('d-none');
        styleImagePreview.querySelector('.upload-placeholder').classList.add('d-none');
        
        // Clear custom style image
        styleImageData = null;
        
        // Update selected style in gallery
        document.querySelectorAll('.style-item').forEach(item => {
            if (item.dataset.styleId === styleId) {
                item.classList.add('selected');
            } else {
                item.classList.remove('selected');
            }
        });
        
        updateTransferButtonState();
    }
    
    /**
     * Update the state of the transfer button based on selected images
     */
    function updateTransferButtonState() {
        if ((contentImageData || contentImage.src) && (styleImageData || selectedStyleId || styleImage.src)) {
            transferButton.disabled = false;
        } else {
            transferButton.disabled = true;
        }
        
        // Hide result and download button when inputs change
        resultImage.classList.add('d-none');
        downloadButton.disabled = true;
        document.querySelector('.result-placeholder').classList.remove('d-none');
    }
    
    /**
     * Apply style transfer to the selected images
     */
    function applyStyleTransfer() {
        // Check if images are selected
        if (!contentImageData && !contentImage.src) {
            alert('Please select a content image');
            return;
        }
        
        if (!styleImageData && !selectedStyleId && !styleImage.src) {
            alert('Please select a style image or choose a predefined style');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.classList.remove('d-none');
        document.querySelector('.result-placeholder').classList.add('d-none');
        resultImage.classList.add('d-none');
        transferButton.disabled = true;
        downloadButton.disabled = true;
        
        // Get parameters
        const method = methodSelect.value;
        const imageSizeValue = parseInt(imageSize.value);
        const styleWeightValue = Math.pow(10, parseInt(styleWeight.value));
        const contentWeightValue = parseFloat(contentWeight.value);
        const iterationsValue = parseInt(iterations.value);
        const alphaValue = parseFloat(alpha.value);
        
        // Prepare form data
        const formData = new FormData();
        
        // Add parameters
        formData.append('method', method);
        formData.append('imageSize', imageSizeValue);
        formData.append('styleWeight', styleWeightValue);
        formData.append('contentWeight', contentWeightValue);
        formData.append('iterations', iterationsValue);
        formData.append('alpha', alphaValue);
        
        // Add images
        if (contentImageData) {
            // Convert data URL to Blob
            const contentBlob = dataURLtoBlob(contentImageData);
            formData.append('content', contentBlob, 'content.jpg');
        }
        
        if (styleImageData) {
            // Convert data URL to Blob
            const styleBlob = dataURLtoBlob(styleImageData);
            formData.append('style', styleBlob, 'style.jpg');
        } else if (selectedStyleId) {
            formData.append('styleId', selectedStyleId);
        }
        
        // Send request to server
        fetch('/api/transfer', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Hide loading spinner
            loadingSpinner.classList.add('d-none');
            transferButton.disabled = false;
            
            if (data.success) {
                // Show result image
                resultImage.src = data.resultUrl;
                resultImage.classList.remove('d-none');
                resultImageUrl = data.resultUrl;
                
                // Enable download button
                downloadButton.disabled = false;
                
                // Show processing time
                console.log(`Processing time: ${data.processingTime} seconds using ${data.method} method`);
            } else {
                alert(`Error: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('Error applying style transfer:', error);
            loadingSpinner.classList.add('d-none');
            transferButton.disabled = false;
            alert('Error applying style transfer. Please try again.');
        });
    }
    
    /**
     * Convert a data URL to a Blob
     * @param {string} dataURL - Data URL to convert
     * @returns {Blob} - Converted Blob
     */
    function dataURLtoBlob(dataURL) {
        const parts = dataURL.split(';base64,');
        const contentType = parts[0].split(':')[1];
        const raw = window.atob(parts[1]);
        const rawLength = raw.length;
        const uInt8Array = new Uint8Array(rawLength);
        
        for (let i = 0; i < rawLength; ++i) {
            uInt8Array[i] = raw.charCodeAt(i);
        }
        
        return new Blob([uInt8Array], { type: contentType });
    }
    
    // Handle drag and drop for content image
    contentImagePreview.addEventListener('dragover', (e) => {
        e.preventDefault();
        contentImagePreview.style.borderColor = '#0d6efd';
    });
    
    contentImagePreview.addEventListener('dragleave', () => {
        contentImagePreview.style.borderColor = '';
    });
    
    contentImagePreview.addEventListener('drop', (e) => {
        e.preventDefault();
        contentImagePreview.style.borderColor = '';
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            
            if (file.type.match('image.*')) {
                const reader = new FileReader();
                
                reader.onload = (event) => {
                    contentImageData = event.target.result;
                    contentImage.src = contentImageData;
                    contentImage.classList.remove('d-none');
                    contentImagePreview.querySelector('.upload-placeholder').classList.add('d-none');
                    updateTransferButtonState();
                };
                
                reader.readAsDataURL(file);
            }
        }
    });
    
    // Handle drag and drop for style image
    styleImagePreview.addEventListener('dragover', (e) => {
        e.preventDefault();
        styleImagePreview.style.borderColor = '#0d6efd';
    });
    
    styleImagePreview.addEventListener('dragleave', () => {
        styleImagePreview.style.borderColor = '';
    });
    
    styleImagePreview.addEventListener('drop', (e) => {
        e.preventDefault();
        styleImagePreview.style.borderColor = '';
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            
            if (file.type.match('image.*')) {
                const reader = new FileReader();
                
                reader.onload = (event) => {
                    styleImageData = event.target.result;
                    styleImage.src = styleImageData;
                    styleImage.classList.remove('d-none');
                    styleImagePreview.querySelector('.upload-placeholder').classList.add('d-none');
                    
                    // Clear selected style
                    selectedStyleId = null;
                    document.querySelectorAll('.style-item.selected').forEach(item => {
                        item.classList.remove('selected');
                    });
                    
                    updateTransferButtonState();
                };
                
                reader.readAsDataURL(file);
            }
        }
    });
});
