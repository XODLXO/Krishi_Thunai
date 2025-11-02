// State variables (Mocked)
let isAuthenticated = false;
let authMode = 'login'; // 'login' or 'register'

// --- 1. CORE NAVIGATION LOGIC ---
const allSections = document.querySelectorAll('.page-section');

function navigateTo(targetId) {
    // 1. Hide all sections
    allSections.forEach(section => {
        // Use the 'hidden' class defined in style.css for smooth transitions
        section.classList.add('hidden');
    });
    
    // 2. Show the target section
    const targetSection = document.getElementById(targetId);
    if (targetSection) {
        targetSection.classList.remove('hidden');
    }

    // 3. Update the active navigation link for visual feedback
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-target') === targetId) {
            link.classList.add('active');
        }
    });
}

// Initialize: Show only the home page on load and attach navigation listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initial load should navigate to 'home' and set it active
    navigateTo('home');
    
    // Attach event listeners to navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = e.target.getAttribute('data-target');
            navigateTo(targetId);
        });
    });
});

// --- 2. MOCK AUTHENTICATION LOGIC ---
const authBtn = document.getElementById('auth-btn');
const authModal = document.getElementById('auth-modal');
const authTitle = document.getElementById('auth-title');
const authSubmitBtn = document.getElementById('auth-submit-btn');
const toggleAuthBtn = document.getElementById('toggle-auth');

function updateAuthUI() {
    if (isAuthenticated) {
        authBtn.textContent = 'Logout';
        authBtn.onclick = logout;
        authBtn.classList.remove('bg-krishi-orange'); // Adjust color for logged in state
        authBtn.classList.add('bg-red-500', 'hover:bg-red-700');
    } else {
        authBtn.textContent = 'Login';
        authBtn.onclick = () => authModal.classList.remove('hidden');
        authBtn.classList.add('bg-krishi-orange'); // Use original orange color
        authBtn.classList.remove('bg-red-500', 'hover:bg-red-700');
    }
}

function logout() {
    isAuthenticated = false;
    // NOTE: Avoid using alert() in final applications, using a custom modal is better.
    alert('Logged out successfully.'); 
    updateAuthUI();
}

// Handle modal state change (login/register)
toggleAuthBtn.addEventListener('click', () => {
    if (authMode === 'login') {
        authMode = 'register';
        authTitle.textContent = 'Register';
        authSubmitBtn.textContent = 'Register';
        toggleAuthBtn.textContent = 'Already have an account? Login';
    } else {
        authMode = 'login';
        authTitle.textContent = 'Login';
        authSubmitBtn.textContent = 'Login';
        toggleAuthBtn.textContent = 'Need an account? Register';
    }
});

// Handle mock form submission
document.getElementById('auth-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const action = authMode === 'login' ? 'Login' : 'Registration';
    // NOTE: Avoid using alert() in final applications, using a custom modal is better.
    alert(`${action} successful! (Mocked)`);
    isAuthenticated = true;
    authModal.classList.add('hidden');
    updateAuthUI();
});

updateAuthUI(); // Initial UI setup

// --- 3. IMAGE UPLOAD & MOCK AI LOGIC ---

// MOCK DATA: Simulating the 13 classes output by your model
const CROP_DISEASES = [
    { crop: "Cauliflower", disease: "Bacterial Spot Rot" },
    { crop: "Cauliflower", disease: "Black Rot" },
    { crop: "Cauliflower", disease: "Downy Mildew" },
    { crop: "Cauliflower", disease: "Healthy" },
    { crop: "Chili", disease: "Anthracnose" },
    { crop: "Chili", disease: "Healthy" },
    { crop: "Chili", disease: "Leaf Curl" },
    { crop: "Chili", disease: "Leaf Spot" },
    { crop: "Chili", disease: "Yellowish" },
    { crop: "Radish", disease: "Black Leaf Spot" },
    { crop: "Radish", disease: "Downy Mildew" },
    { crop: "Radish", disease: "Healthy" },
    { crop: "Radish", disease: "Mosaic" },
];

// Function to handle image preview
document.getElementById('cropImage').addEventListener('change', function(e) {
    const preview = document.getElementById('preview-img');
    const previewContainer = document.getElementById('image-preview');
    if (e.target.files && e.target.files[0]) {
        preview.src = URL.createObjectURL(e.target.files[0]);
        previewContainer.classList.remove('hidden');
    } else {
        previewContainer.classList.add('hidden');
    }
});

// Core MOCK Prediction Function
async function mockPrediction(file) {
    // Simulate network delay and model processing (3 seconds)
    await new Promise(resolve => setTimeout(resolve, 3000)); 

    // --- MOCK LOGIC: Randomly select a disease and confidence ---
    const randomResult = CROP_DISEASES[Math.floor(Math.random() * CROP_DISEASES.length)];
    const confidence = (Math.random() * 0.2 + 0.75); // Generates 75% to 95% confidence
    const severity = (Math.random() * 4 + 3).toFixed(1); // Generates Severity score 3.0 to 7.0

    // This is the structure you would expect from your real backend API
    return {
        crop: randomResult.crop,
        disease: randomResult.disease,
        confidence: confidence,
        severity_score: severity + '/10' 
    };
}

// Handle form submission for analysis
document.getElementById('analysisForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const fileInput = document.getElementById('cropImage');
    const statusText = document.getElementById('analysis-status');
    const resultsDiv = document.getElementById('analysis-results');
    const analyzeBtn = document.getElementById('analyzeBtn');

    if (fileInput.files.length === 0) {
        statusText.textContent = "Please select an image file.";
        return;
    }

    // 1. Update UI for Loading
    statusText.textContent = 'Analyzing image... Please wait (Mocking 3 seconds).';
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Analyzing...';
    resultsDiv.classList.add('hidden');

    try {
        // 2. Run Mock Prediction (***Replace this line with your real API call later***)
        // Example of what a real API call would look like:
        // const response = await fetch('YOUR_AWS_FLASK_API_ENDPOINT', { method: 'POST', body: new FormData(this) });
        // const data = await response.json();
        const data = await mockPrediction(fileInput.files[0]);

        // 3. Update UI with Results
        document.getElementById('res-crop').textContent = data.crop;
        document.getElementById('res-prediction').textContent = data.disease;
        document.getElementById('res-confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
        document.getElementById('res-severity').textContent = data.severity_score;

        resultsDiv.classList.remove('hidden');
        statusText.textContent = 'Analysis Complete!';

    } catch (error) {
        statusText.textContent = `Error during analysis: ${error.message}`;
        console.error("Prediction error:", error);
    } finally {
        // 4. Reset UI
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-brain mr-2"></i> Analyze Crop Health';
    }
});
