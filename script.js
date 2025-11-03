document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload');
    const imagePreviewContainer = document.getElementById('image-preview');
    const previewImage = document.getElementById('preview-image');

    // Function to handle image preview
    imageUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                imagePreviewContainer.style.display = 'block';
            }
            reader.readAsDataURL(file);

            // In a later step, we will call the API here:
            // uploadAndPredict(file); 
        } else {
            imagePreviewContainer.style.display = 'none';
            previewImage.src = '';
        }
    });

    // Placeholder for the prediction function (to be implemented in Phase 3)
    // async function uploadAndPredict(file) {
    //     const resultArea = document.getElementById('result-area');
    //     // ... API call logic goes here
    //     resultArea.style.display = 'block';
    // }
});
