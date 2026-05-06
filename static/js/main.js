document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const resultImage = document.getElementById('result-image');
    const resultCount = document.getElementById('result-count');

    // Handle Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            handleUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleUpload(this.files[0]);
        }
    });

    function handleUpload(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }

        const formData = new FormData();
        formData.append('image', file);

        // Show Loader
        loader.style.display = 'flex';
        resultsSection.style.display = 'none';

        fetch('/api/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                resultImage.src = data.annotated_image;
                resultCount.textContent = `${data.count} Holes Detected`;
                resultsSection.style.display = 'block';
                // Scroll to results
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during detection.');
        })
        .finally(() => {
            loader.style.display = 'none';
        });
    }
});
