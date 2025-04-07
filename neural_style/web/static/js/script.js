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
    const processingTimeContainer = document.getElementById('processingTimeContainer');
    const processingTime = document.getElementById('processingTime');
    
    const styleGallery = document.getElementById('styleGallery');
    const noStylesMessage = document.getElementById('noStylesMessage');
    
    const transferButton = document.getElementById('transferButton');
    
    const methodSelect = document.getElementById('methodSelect');
    const imageSizeRange = document.getElementById('imageSizeRange');
    const imageSizeValue = document.getElementById('imageSizeValue');
    const styleWeightRange = document.getElementById('styleWeightRange');
    const styleWeightValue = document.getElementById('styleWeightValue');
    const contentWeightRange = document.getElementById('contentWeightRange');
    const contentWeightValue = document.getElementById('contentWeightValue');
    
    // State
    let contentImageData = null;
    let styleImageData = null;
    let selectedStyleId = null;
    
    // Initialize
    loadPredefinedStyles();
    setupEventListeners();
    
    /**
     * Load predefined styles from the server
     */
    function loadPredefinedStyles() {
        fetch('/api/styles')
            .then(response => response.json())
            .then(styles => {
                if (styles.length > 0) {
                    noStylesMessage.classList.add('d-none');
                    styles.forEach(style => {
                        const styleItem = document.createElement('div');
                        styleItem.className = 'style-item';
                        styleItem.dataset.styleId = style.id;
                        
                        const styleImg = document.createElement('img');
                        styleImg.src = style.path;
                        styleImg.alt = style.name;
                        styleImg.title = style.name;
                        
                        styleItem.appendChild(styleImg);
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
                    document.querySelectorAll('.style-item').forEach(item => {
                        item.classList.remove('selected');
                    });
                    
                    updateTransferButtonState();
                };
                
                reader.readAsDataURL(file);
            }
        });
        
        // Transfer button
        transferButton.addEventListener('click', applyStyleTransfer);
        
        // Range inputs
        imageSizeRange.addEventListener('input', () => {
            imageSizeValue.textContent = imageSizeRange.value;
        });
        
        styleWeightRange.addEventListener('input', () => {
            const value = Math.pow(10, parseFloat(styleWeightRange.value));
            styleWeightValue.textContent = Math.round(value);
        });
        
        contentWeightRange.addEventListener('input', () => {
            contentWeightValue.textContent = contentWeightRange.value;
        });
        
        // Method select
        methodSelect.addEventListener('change', () => {
            if (methodSelect.value === 'fast') {
                // Disable style weight and content weight for fast method
                styleWeightRange.disabled = true;
                contentWeightRange.disabled = true;
            } else {
                // Enable style weight and content weight for other methods
                styleWeightRange.disabled = false;
                contentWeightRange.disabled = false;
            }
        });
    }
    
    /**
     * Select a predefined style
     * @param {string} styleId - ID of the selected style
     * @param {string} stylePath - Path to the style image
     */
    function selectPredefinedStyle(styleId, stylePath) {
        // Update selected style
        selectedStyleId = styleId;
        styleImageData = null;
        
        // Update UI
        styleImage.src = stylePath;
        styleImage.classList.remove('d-none');
        styleImagePreview.querySelector('.upload-placeholder').classList.add('d-none');
        
        // Update style gallery
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
        if (contentImageData && (styleImageData || selectedStyleId)) {
            transferButton.disabled = false;
        } else {
            transferButton.disabled = true;
        }
    }
    
    /**
     * Apply style transfer to the selected images
     */
    function applyStyleTransfer() {
        // Check if content and style images are selected
        if (!contentImageData || (!styleImageData && !selectedStyleId)) {
            alert('Please select both content and style images');
            return;
        }
        
        // Show loading spinner
        loadingSpinner.classList.remove('d-none');
        resultImage.classList.add('d-none');
        transferButton.disabled = true;
        processingTimeContainer.classList.add('d-none');
        
        // Get parameters
        const method = methodSelect.value;
        const imageSize = parseInt(imageSizeRange.value);
        const styleWeight = Math.pow(10, parseFloat(styleWeightRange.value));
        const contentWeight = parseFloat(contentWeightRange.value);
        
        // Create request data
        const requestData = {
            contentImage: contentImageData,
            method: method,
            imageSize: imageSize,
            styleWeight: styleWeight,
            contentWeight: contentWeight
        };
        
        // Add style image or style ID
        if (styleImageData) {
            requestData.styleImage = styleImageData;
        } else if (selectedStyleId) {
            requestData.styleId = selectedStyleId;
        }
        
        // Send request to API
        fetch('/api/transfer-base64', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Show result
                resultImage.src = data.resultImage || data.resultUrl;
                resultImage.classList.remove('d-none');
                
                // Show processing time
                processingTime.textContent = data.processingTime;
                processingTimeContainer.classList.remove('d-none');
                
                // Add download button
                addDownloadButton(data.resultImage || data.resultUrl);
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error applying style transfer:', error);
            alert('An error occurred while applying style transfer');
        })
        .finally(() => {
            // Hide loading spinner
            loadingSpinner.classList.add('d-none');
            transferButton.disabled = false;
        });
    }
    
    /**
     * Add a download button to the result image
     * @param {string} imageUrl - URL or data URL of the result image
     */
    function addDownloadButton(imageUrl) {
        // Remove existing download button
        const existingButton = document.querySelector('.download-btn');
        if (existingButton) {
            existingButton.remove();
        }
        
        // Create download button
        const downloadButton = document.createElement('button');
        downloadButton.className = 'download-btn';
        downloadButton.innerHTML = '<i class="fas fa-download"></i>';
        downloadButton.title = 'Download Image';
        
        // Add click event
        downloadButton.addEventListener('click', (e) => {
            e.stopPropagation();
            
            // Create a temporary link
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = 'styled_image.jpg';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
        
        // Add button to result container
        document.getElementById('resultImageContainer').querySelector('.image-preview').appendChild(downloadButton);
    }
    
    // Handle drag and drop for content image
    contentImagePreview.addEventListener('dragover', (e) => {
        e.preventDefault();
        contentImagePreview.style.borderColor = '#0d6efd';
    });
    
    contentImagePreview.addEventListener('dragleave', () => {
        contentImagePreview.style.borderColor = '#dee2e6';
    });
    
    contentImagePreview.addEventListener('drop', (e) => {
        e.preventDefault();
        contentImagePreview.style.borderColor = '#dee2e6';
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.match('image.*')) {
                contentImageInput.files = e.dataTransfer.files;
                
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
        styleImagePreview.style.borderColor = '#dee2e6';
    });
    
    styleImagePreview.addEventListener('drop', (e) => {
        e.preventDefault();
        styleImagePreview.style.borderColor = '#dee2e6';
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.match('image.*')) {
                styleImageInput.files = e.dataTransfer.files;
                
                const reader = new FileReader();
                reader.onload = (event) => {
                    styleImageData = event.target.result;
                    styleImage.src = styleImageData;
                    styleImage.classList.remove('d-none');
                    styleImagePreview.querySelector('.upload-placeholder').classList.add('d-none');
                    
                    // Clear selected style
                    selectedStyleId = null;
                    document.querySelectorAll('.style-item').forEach(item => {
                        item.classList.remove('selected');
                    });
                    
                    updateTransferButtonState();
                };
                
                reader.readAsDataURL(file);
            }
        }
    });
});
