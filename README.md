<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Krishi Thunai - The Farmer's Companion</title>
    <!-- Load Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Font Awesome for Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'krishi-green': '#22C55E', // Vibrant Green
                        'krishi-orange': '#F97316', // Vibrant Orange
                        'krishi-yellow': '#FBBF24', // Sunny Yellow
                        'krishi-earth': '#78350F', // Earthy Brown
                        'krishi-bg': '#F3F4F6', // Light Gray BG
                    },
                    fontFamily: {
                        sans: ['Inter', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <style>
        /* Custom Styles for Background Image and Fixed Elements */
        #home {
            background-image: url('https://placehold.co/1920x1080/047857/ffffff?text=Krishi+Thunai+Front+Page+Background');
            background-size: cover;
            background-position: center;
        }

        /* Ensure smooth transitions */
        .transition-all {
            transition-property: all;
            transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
            transition-duration: 300ms;
        }

        /* Adjust body to prevent content from hiding under fixed header */
        #app-container {
            padding-top: 64px; /* Height of the header */
        }

        /* Custom scrollbar for chatbot body */
        #chatbot-body-content {
            overflow-y: auto;
        }
        #chatbot-body-content::-webkit-scrollbar {
            width: 6px;
        }
        #chatbot-body-content::-webkit-scrollbar-thumb {
            background-color: rgba(34, 197, 94, 0.5);
            border-radius: 3px;
        }
    </style>
</head>
<body class="bg-krishi-bg font-sans">

    <!-- ----------------------- Sidebar Navigation ----------------------- -->
    <aside id="sidebar" class="fixed top-0 right-0 h-full w-0 bg-krishi-earth text-white overflow-x-hidden z-[200] shadow-xl transition-all duration-300 md:w-0">
        <button class="absolute top-4 right-6 text-4xl leading-none bg-transparent border-none cursor-pointer hover:text-krishi-yellow transition-all" onclick="toggleSidebar()">&times;</button>
        <div class="p-6 pt-16 text-center border-b border-krishi-green/50 mb-4">
            <img src="https://placehold.co/80x80/22C55E/ffffff?text=Logo" alt="Krishi Thunai Logo" class="w-20 h-20 rounded-full mx-auto mb-2 shadow-lg">
            <h3 class="text-xl font-bold">Krishi Thunai</h3>
        </div>
        <nav class="flex flex-col">
            <a href="#" onclick="navigateTo('home')" class="p-4 flex items-center hover:bg-krishi-green transition-all"><i class="fas fa-home w-6 mr-3"></i> Home</a>
            <a href="#" onclick="navigateTo('features')" class="p-4 flex items-center hover:bg-krishi-green transition-all"><i class="fas fa-seedling w-6 mr-3"></i> Features</a>
            <a href="#" onclick="navigateTo('dashboard')" class="p-4 flex items-center hover:bg-krishi-green transition-all"><i class="fas fa-chart-line w-6 mr-3"></i> Dashboard</a>
            <a href="#" onclick="navigateTo('registration')" class="p-4 flex items-center hover:bg-krishi-green transition-all"><i class="fas fa-user-plus w-6 mr-3"></i> Registration</a>
            <a href="#" onclick="navigateTo('login-page')" class="p-4 flex items-center hover:bg-krishi-green transition-all"><i class="fas fa-sign-in-alt w-6 mr-3"></i> Login</a>
            <a href="#" onclick="navigateTo('about')" class="p-4 flex items-center hover:bg-krishi-green transition-all"><i class="fas fa-info-circle w-6 mr-3"></i> About</a>
            <a href="#" onclick="navigateTo('contact')" class="p-4 flex items-center hover:bg-krishi-green transition-all"><i class="fas fa-envelope w-6 mr-3"></i> Contact</a>
        </nav>
        <div class="absolute bottom-4 left-0 w-full text-center text-sm text-gray-300">
            &copy; 2024 Krishi Thunai
        </div>
    </aside>

    <!-- ----------------------- Main Header/Navbar ----------------------- -->
    <header class="fixed top-0 left-0 w-full bg-krishi-green text-white shadow-lg z-100 flex justify-between items-center px-4 py-3">
        <div class="flex items-center">
            <img src="https://placehold.co/40x40/FFFFFF/22C55E?text=K" alt="Krishi Thunai Logo" class="w-10 h-10 rounded-full mr-3">
            <h1 class="text-xl font-bold">Krishi Thunai</h1>
        </div>
        <!-- Hamburger Menu Button -->
        <button class="text-3xl hover:text-krishi-yellow transition-all" onclick="toggleSidebar()">
            &#9776;
        </button>
    </header>

    <!-- ----------------------- Main Content Sections ----------------------- -->
    <main id="app-container">

        <!-- 1. Home/Landing Page (Default View) -->
        <section id="home" class="page active h-screen flex justify-center items-center relative p-4">
            <div class="absolute inset-0 bg-black/50"></div>
            <div class="relative text-white text-center p-8 max-w-4xl backdrop-blur-sm bg-black/20 rounded-xl shadow-2xl">
                <img src="https://placehold.co/150x150/FFFFFF/22C55E?text=LOGO" alt="Krishi Thunai Logo" class="w-36 h-36 rounded-full mx-auto mb-6 shadow-xl border-4 border-krishi-yellow transition-all hover:scale-105">
                <h2 class="text-5xl md:text-6xl font-extrabold mb-4 drop-shadow-lg">Krishi Thunai</h2>
                <p class="text-xl md:text-2xl font-light mb-8 italic">Your AI-Powered Companion for Sustainable and Profitable Farming.</p>
                <button class="btn-primary" onclick="navigateTo('login-page')">Get Started / Login</button>
            </div>
        </section>

        <!-- 2. Login Page -->
        <section id="login-page" class="page flex justify-center items-center py-10 md:py-20">
            <div class="w-full max-w-md p-8 bg-white rounded-xl shadow-2xl border-t-8 border-krishi-green">
                <h2 class="text-3xl font-bold text-center text-krishi-green mb-6">User Login</h2>
                <form id="loginForm" class="space-y-4">
                    <div>
                        <label for="username" class="input-label"><i class="fas fa-user mr-2"></i> Username</label>
                        <input type="text" id="username" class="input-field" placeholder="Enter your username" required>
                    </div>
                    <div>
                        <label for="password" class="input-label"><i class="fas fa-lock mr-2"></i> Password</label>
                        <input type="password" id="password" class="input-field" placeholder="Enter your password" required>
                    </div>
                    <button type="submit" class="btn-primary w-full mt-6">Login</button>
                    <p class="text-sm text-center text-red-500 font-semibold mt-2" id="login-error"></p>
                </form>
                <p class="text-center text-sm mt-4">New user? <a href="#" onclick="navigateTo('registration')" class="text-krishi-orange hover:underline font-semibold">Register here</a></p>
            </div>
        </section>

        <!-- 3. Main Dashboard (Hidden until successful login) -->
        <section id="dashboard" class="page hidden-page p-6 md:p-10">
            <h2 class="text-4xl font-extrabold text-krishi-earth mb-8 border-b-4 border-krishi-yellow pb-2">👋 Welcome Back, Farmer!</h2>
            <p class="text-lg text-gray-700 mb-8">Access your personalized farming tools below. Click on a tile to explore the module.</p>
            
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <div class="dashboard-tile bg-krishi-green" onclick="navigateTo('crop-analysis')">
                    <i class="fas fa-leaf text-4xl mb-3"></i>
                    <h3 class="text-xl font-bold">AI Crop Analysis</h3>
                    <p class="text-sm mt-1">Detect diseases and get health scores instantly.</p>
                </div>
                <div class="dashboard-tile bg-krishi-orange" onclick="navigateTo('weather')">
                    <i class="fas fa-cloud-sun-rain text-4xl mb-3"></i>
                    <h3 class="text-xl font-bold">Weather & Irrigation</h3>
                    <p class="text-sm mt-1">Smart guidance based on real-time climate data.</p>
                </div>
                <div class="dashboard-tile bg-krishi-yellow text-krishi-earth" onclick="navigateTo('market')">
                    <i class="fas fa-chart-line text-4xl mb-3"></i>
                    <h3 class="text-xl font-bold">Market Prices</h3>
                    <p class="text-sm mt-1">Real-time Mandi rates and trend analysis.</p>
                </div>
                <div class="dashboard-tile bg-krishi-earth" onclick="navigateTo('management')">
                    <i class="fas fa-calendar-alt text-4xl mb-3"></i>
                    <h3 class="text-xl font-bold">Farm Management</h3>
                    <p class="text-sm mt-1">Crop calendar, reminders, and expense tracker.</p>
                </div>
            </div>

            <button class="btn-secondary mt-10" onclick="logout()">Logout</button>
        </section>

        <!-- 4. Registration Page -->
        <section id="registration" class="page flex justify-center items-center py-10 md:py-20">
            <div class="w-full max-w-md p-8 bg-white rounded-xl shadow-2xl border-t-8 border-krishi-orange">
                <h2 class="text-3xl font-bold text-center text-krishi-orange mb-6">New Farmer Registration</h2>
                <form id="registrationForm" class="space-y-4">
                    <div>
                        <label for="reg-name" class="input-label"><i class="fas fa-address-card mr-2"></i> Full Name</label>
                        <input type="text" id="reg-name" class="input-field" required>
                    </div>
                    <div>
                        <label for="reg-username" class="input-label"><i class="fas fa-user mr-2"></i> Username</label>
                        <input type="text" id="reg-username" class="input-field" required>
                    </div>
                    <div>
                        <label for="reg-password" class="input-label"><i class="fas fa-lock mr-2"></i> Password</label>
                        <input type="password" id="reg-password" class="input-field" required>
                    </div>
                    <div>
                        <label for="reg-confirm-password" class="input-label"><i class="fas fa-lock mr-2"></i> Confirm Password</label>
                        <input type="password" id="reg-confirm-password" class="input-field" required>
                    </div>
                    <button type="submit" class="btn-secondary w-full mt-6">Register Account</button>
                    <p class="text-sm text-center text-red-500 font-semibold mt-2" id="reg-error"></p>
                </form>
            </div>
        </section>

        <!-- 5. Features Page -->
        <section id="features" class="page p-6 md:p-10">
            <h2 class="text-4xl font-extrabold text-krishi-green mb-6">🌱 Krishi Thunai Key Features</h2>
            <div class="space-y-6 max-w-3xl">
                <div class="p-4 bg-white rounded-lg shadow-md border-l-4 border-krishi-green">
                    <h3 class="text-xl font-semibold text-krishi-earth"><i class="fas fa-robot mr-2"></i> AI Crop & Disease Analysis (Stage 2)</h3>
                    <p class="mt-1 text-gray-600">Upload an image of your crop to receive instant diagnosis, confidence score, and recommended treatment plan.</p>
                </div>
                <div class="p-4 bg-white rounded-lg shadow-md border-l-4 border-krishi-orange">
                    <h3 class="text-xl font-semibold text-krishi-earth"><i class="fas fa-cloud-sun-rain mr-2"></i> Weather & Irrigation Module (Stage 3)</h3>
                    <p class="mt-1 text-gray-600">Real-time weather data and soil-specific irrigation recommendations to optimize water usage.</p>
                </div>
                <div class="p-4 bg-white rounded-lg shadow-md border-l-4 border-krishi-yellow">
                    <h3 class="text-xl font-semibold text-krishi-earth"><i class="fas fa-comments mr-2"></i> Multilingual Smart Chatbot (Stage 4)</h3>
                    <p class="mt-1 text-gray-600">Get instant advice and answers to farming questions in English, Tamil, or Hindi.</p>
                </div>
                <div class="p-4 bg-white rounded-lg shadow-md border-l-4 border-krishi-earth">
                    <h3 class="text-xl font-semibold text-krishi-earth"><i class="fas fa-tags mr-2"></i> Market Intelligence (Stage 5)</h3>
                    <p class="mt-1 text-gray-600">Check nearest mandi prices, historical trends, and receive ML-based sell recommendations.</p>
                </div>
            </div>
        </section>

        <!-- 6. Placeholder Pages (About, Contact, Specific Feature Modules) -->
        <section id="crop-analysis" class="page p-6 md:p-10">
            <h2 class="text-4xl font-extrabold text-krishi-green mb-6">🔬 AI Crop Analysis</h2>
            <div class="bg-white p-6 rounded-lg shadow-lg max-w-3xl">
                <p class="text-krishi-earth font-semibold mb-4">Upload a clear photo of the leaf or plant part for instant diagnosis.</p>
                <input type="file" id="cropImage" accept="image/*" class="w-full mb-4 p-2 border border-gray-300 rounded-lg">
                <button class="btn-primary" onclick="simulateAnalysis()">Analyze Crop</button>
                
                <div id="analysis-results" class="mt-6 p-4 bg-krishi-bg rounded-lg border-l-4 border-krishi-orange hidden">
                    <h3 class="text-xl font-bold text-krishi-orange mb-2">Analysis Result (Mock)</h3>
                    <p><strong>Prediction:</strong> <span id="mock-prediction"></span></p>
                    <p><strong>Confidence:</strong> <span id="mock-confidence"></span></p>
                    <p><strong>Severity:</strong> <span id="mock-severity"></span></p>
                    <button class="btn-secondary mt-3">Download PDF Report</button>
                </div>
            </div>
        </section>

        <section id="weather" class="page p-6 md:p-10">
            <h2 class="text-4xl font-extrabold text-krishi-orange mb-6">🌧️ Weather & Irrigation Module</h2>
            <div class="bg-white p-6 rounded-lg shadow-lg max-w-3xl">
                <input type="text" id="weatherLocation" placeholder="Enter your city or zip code" class="input-field mb-4">
                <button class="btn-primary" onclick="simulateWeather()">Get Forecast & Irrigation Advice</button>
                
                <div id="weather-results" class="mt-6 p-4 bg-krishi-bg rounded-lg border-l-4 border-krishi-green hidden">
                    <h3 class="text-xl font-bold text-krishi-green mb-2">Weather Report (Mock)</h3>
                    <p id="mock-weather"></p>
                </div>
            </div>
        </section>

        <section id="market" class="page p-6 md:p-10">
            <h2 class="text-4xl font-extrabold text-krishi-yellow mb-6 text-krishi-earth">📈 Market Intelligence</h2>
            <div class="bg-white p-6 rounded-lg shadow-lg max-w-3xl">
                <select id="marketCommodity" class="input-field mb-4">
                    <option value="">-- Select Commodity --</option>
                    <option value="Paddy">Paddy (Rice)</option>
                    <option value="Wheat">Wheat</option>
                    <option value="Tomato">Tomato</option>
                    <option value="Cotton">Cotton</option>
                </select>
                <button class="btn-primary w-full" onclick="simulateMarket()">Check Mandi Price</button>
                
                <div id="market-results" class="mt-6 p-4 bg-krishi-bg rounded-lg border-l-4 border-krishi-yellow hidden">
                    <h3 class="text-xl font-bold text-krishi-earth mb-2">Market Data (Mock)</h3>
                    <p id="mock-market"></p>
                </div>
            </div>
        </section>

        <section id="management" class="page p-6 md:p-10">
            <h2 class="text-4xl font-extrabold text-krishi-earth mb-6">📅 Farm Management & Calendar</h2>
            <div class="bg-white p-6 rounded-lg shadow-lg max-w-3xl">
                <p class="text-lg text-gray-700">This module will feature your crop-wise calendar, planting reminders, expense tracking, and yield log. (Placeholder content)</p>
                <div class="mt-4 p-4 bg-krishi-bg rounded-lg border-l-4 border-krishi-earth">
                    <p class="font-bold">Next Task Reminder:</p>
                    <p class="text-sm">Apply fertilizer (Urea) to Paddy field on **Nov 15, 2024**.</p>
                </div>
            </div>
        </section>

        <section id="about" class="page p-6 md:p-10">
            <h2 class="text-4xl font-extrabold text-krishi-green mb-6">💡 About Krishi Thunai</h2>
            <div class="text-lg space-y-4 max-w-3xl">
                <p>Krishi Thunai (meaning 'Agriculture Companion' in Tamil) is a revolutionary platform dedicated to supporting farmers through modern technology.</p>
                <p>We leverage Artificial Intelligence, real-time weather data, and market intelligence to provide actionable, timely advice, leading to higher yields and better profitability.</p>
            </div>
        </section>

        <section id="contact" class="page p-6 md:p-10">
            <h2 class="text-4xl font-extrabold text-krishi-orange mb-6">📞 Contact Us</h2>
            <div class="text-lg space-y-4 max-w-3xl">
                <p><i class="fas fa-envelope mr-3 text-krishi-green"></i> **Email:** support@krishithunai.com</p>
                <p><i class="fas fa-phone mr-3 text-krishi-green"></i> **Phone:** +91 98765 43210</p>
                <p><i class="fas fa-map-marker-alt mr-3 text-krishi-green"></i> **Address:** AgTech Innovation Hub, Chennai, India.</p>
            </div>
        </section>

    </main>

    <!-- ----------------------- Embedded Chatbot (Bottom-Right) ----------------------- -->
    <div id="embedded-chatbot" class="fixed bottom-4 right-4 w-11/12 max-w-xs h-12 bg-white rounded-xl shadow-2xl overflow-hidden transition-all duration-500 z-50">
        <div class="chatbot-header bg-krishi-green text-white p-3 flex justify-between items-center cursor-pointer" onclick="toggleChatbot()">
            <span class="font-bold"><i class="fas fa-comments mr-2"></i> Smart Assistant</span>
            <span class="text-xl font-bold" id="chatbot-toggle-icon">+</span>
        </div>
        <div class="chatbot-body h-[320px] p-3 flex flex-col justify-end" id="chatbot-body-content">
            <!-- Messages will appear here -->
            <div class="text-center text-sm text-gray-500 mb-2">How can I assist your farming today?</div>
        </div>
        <div class="chatbot-footer border-t border-gray-200 p-2 flex items-center">
            <input type="text" id="chatbot-input" placeholder="Type message in English, Tamil, or Hindi..." class="flex-grow p-2 border border-gray-300 rounded-l-lg focus:outline-none focus:border-krishi-green">
            <button id="chatbot-send-btn" class="bg-krishi-orange text-white p-2 rounded-r-lg hover:bg-krishi-earth transition-all"><i class="fas fa-paper-plane"></i></button>
        </div>
    </div>
    
    <!-- Custom Tailwind Classes for Reuse -->
    <style>
        .page { display: none; min-height: calc(100vh - 64px); }
        .page.active { display: block; }
        .hidden-page { display: none !important; }
        .input-label { display: block; margin-bottom: 5px; font-weight: 600; color: #78350F; }
        .input-field { width: 100%; padding: 10px; border: 1px solid #D1D5DB; border-radius: 8px; transition: border-color 0.3s, box-shadow 0.3s; }
        .input-field:focus { outline: none; border-color: #F97316; box-shadow: 0 0 0 2px rgba(249, 115, 22, 0.2); }
        
        .btn-primary { 
            background-color: #F97316; color: white; padding: 12px 30px; border-radius: 8px; font-weight: bold; cursor: pointer;
            transition: all 0.3s; box-shadow: 0 4px #78350F;
        }
        .btn-primary:hover { background-color: #EA580C; transform: translateY(-2px); box-shadow: 0 6px #78350F; }
        
        .btn-secondary {
            background-color: #22C55E; color: white; padding: 12px 30px; border-radius: 8px; font-weight: bold; cursor: pointer;
            transition: all 0.3s; box-shadow: 0 4px #15803D;
        }
        .btn-secondary:hover { background-color: #16A34A; transform: translateY(-2px); box-shadow: 0 6px #15803D; }

        .dashboard-tile {
            @apply p-6 rounded-xl text-white shadow-xl flex flex-col items-center justify-center cursor-pointer transition-all hover:scale-[1.02] hover:shadow-2xl;
        }
        
        /* Mobile adjustment for sidebar */
        @media (max-width: 768px) {
            #sidebar.open {
                width: 75%;
            }
        }
    </style>

    <script>
        // --- Global State ---
        let isLoggedIn = false;
        const pages = document.querySelectorAll('.page');
        const sidebar = document.getElementById('sidebar');
        const loginPage = document.getElementById('login-page');
        const dashboardPage = document.getElementById('dashboard');
        
        // --- Navigation and Routing ---
        function navigateTo(pageId) {
            // 1. Close sidebar
            if (sidebar.classList.contains('open')) {
                toggleSidebar();
            }

            // 2. Auth check for Dashboard
            if (pageId === 'dashboard' && !isLoggedIn) {
                alert('Please log in to view the dashboard.');
                pageId = 'login-page';
            }
            
            // 3. Hide all pages and show the target page
            pages.forEach(page => {
                page.classList.remove('active');
            });
            
            const targetPage = document.getElementById(pageId);
            if (targetPage) {
                targetPage.classList.add('active');
            } else {
                console.error(`Page ID '${pageId}' not found.`);
                navigateTo('home');
            }
        }

        // --- Sidebar Toggle ---
        function toggleSidebar() {
            const isOpen = sidebar.classList.toggle('open');
            sidebar.style.width = isOpen ? (window.innerWidth < 768 ? '75%' : '300px') : '0';
        }

        // --- Stage 1: Login Logic ---
        document.getElementById('loginForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const errorElement = document.getElementById('login-error');

            // Mock Authentication
            if (username === 'farmer' && password === 'krishi123') {
                isLoggedIn = true;
                errorElement.textContent = '';
                document.getElementById('loginForm').reset();
                navigateTo('dashboard');
            } else {
                errorElement.textContent = 'Invalid credentials. Use farmer/krishi123.';
                setTimeout(() => errorElement.textContent = '', 4000);
            }
        });

        function logout() {
            isLoggedIn = false;
            navigateTo('login-page');
            alert('You have been successfully logged out.');
        }

        // --- Registration Logic ---
        document.getElementById('registrationForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const password = document.getElementById('reg-password').value;
            const confirmPassword = document.getElementById('reg-confirm-password').value;
            const errorElement = document.getElementById('reg-error');

            if (password !== confirmPassword) {
                errorElement.textContent = 'Passwords do not match.';
                return;
            }

            // Mock successful registration
            errorElement.textContent = 'Registration successful! Please log in.';
            document.getElementById('registrationForm').reset();
            setTimeout(() => {
                errorElement.textContent = '';
                navigateTo('login-page');
            }, 3000);
        });

        // --- Embedded Chatbot Toggle ---
        const embeddedChatbot = document.getElementById('embedded-chatbot');
        const chatbotBody = document.getElementById('chatbot-body-content');
        const toggleIcon = document.getElementById('chatbot-toggle-icon');

        function toggleChatbot() {
            const isOpen = embeddedChatbot.classList.toggle('h-12');
            embeddedChatbot.classList.toggle('h-[400px]');
            toggleIcon.textContent = isOpen ? '+' : '-';
            
            if (!isOpen) {
                chatbotBody.scrollTop = chatbotBody.scrollHeight; // Scroll to bottom when opening
            }
        }
        
        // Start closed initially
        embeddedChatbot.classList.add('h-12');


        // --- Stage 4: Chatbot Mock Logic ---
        const chatInput = document.getElementById('chatbot-input');
        const chatSendBtn = document.getElementById('chatbot-send-btn');

        function addMessage(text, sender = 'user') {
            const msgDiv = document.createElement('div');
            msgDiv.className = `p-3 mb-2 rounded-xl max-w-[85%] shadow-md text-sm transition-all ${
                sender === 'user' 
                ? 'bg-krishi-orange text-white self-end ml-auto rounded-br-none' 
                : 'bg-krishi-green text-white self-start mr-auto rounded-tl-none'
            }`;
            msgDiv.textContent = text;
            chatbotBody.appendChild(msgDiv);
            chatbotBody.scrollTop = chatbotBody.scrollHeight;
        }

        function getChatbotResponse(userMessage) {
            const msg = userMessage.toLowerCase();
            let response = "";

            if (msg.includes("weather") || msg.includes("rain")) {
                response = "I see you're checking the weather! Please use the **Weather & Irrigation Module** for real-time forecast and smart watering advice based on your location.";
            } else if (msg.includes("disease") || msg.includes("health")) {
                response = "For quick diagnosis, use the **AI Crop Analysis** feature! Upload a leaf image, and I'll give you the prediction and severity score.";
            } else if (msg.includes("price") || msg.includes("mandi")) {
                response = "Check the **Market Intelligence** module for the latest commodity prices in your nearest mandi and get personalized selling advice.";
            } else if (msg.includes("வணக்கம்") || msg.includes("tamil") || msg.includes("tamil")) {
                response = "வணக்கம்! நான் விவசாயிகளுக்கு உதவும் ஒரு ஸ்மார்ட் உதவியாளர். நான் உங்களுக்கு என்ன உதவி செய்ய முடியும்?";
            } else {
                response = "Hello! I'm Krishi Thunai, your farm assistant. I can guide you to our AI tools for weather, disease analysis, and market prices. Ask me about a specific feature!";
            }
            
            setTimeout(() => addMessage(response, 'bot'), 1000);
        }

        function handleChatInput() {
            const userMessage = chatInput.value.trim();
            if (userMessage === '') return;

            addMessage(userMessage, 'user');
            getChatbotResponse(userMessage);
            chatInput.value = '';
        }

        chatSendBtn.addEventListener('click', handleChatInput);
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') handleChatInput();
        });


        // --- Stage 2: Crop Analysis Mock Logic ---
        function simulateAnalysis() {
            const fileInput = document.getElementById('cropImage');
            const resultsDiv = document.getElementById('analysis-results');
            
            if (fileInput.files.length === 0) {
                alert('Please select an image file first!');
                return;
            }

            // Simulate loading
            resultsDiv.classList.add('hidden');
            document.getElementById('mock-prediction').textContent = 'Analyzing...';
            document.getElementById('mock-confidence').textContent = 'Loading...';
            document.getElementById('mock-severity').textContent = 'Loading...';
            
            setTimeout(() => {
                const fileName = fileInput.files[0].name.toLowerCase();
                let prediction, confidence, severity;

                if (fileName.includes('healthy') || fileName.includes('good')) {
                    prediction = 'Healthy Crop (Paddy)';
                    confidence = '99.1%';
                    severity = 'None';
                } else if (fileName.includes('blight') || fileName.includes('disease')) {
                    prediction = 'Bacterial Leaf Blight';
                    confidence = '85.4%';
                    severity = 'High - Immediate fungicide application recommended.';
                } else {
                    prediction = 'Unknown Pest/Disease';
                    confidence = '70.0%';
                    severity = 'Medium - Retake image or contact expert.';
                }

                document.getElementById('mock-prediction').textContent = prediction;
                document.getElementById('mock-confidence').textContent = confidence;
                document.getElementById('mock-severity').textContent = severity;
                resultsDiv.classList.remove('hidden');
            }, 2000);
        }

        // --- Stage 3: Weather Mock Logic ---
        function simulateWeather() {
            const location = document.getElementById('weatherLocation').value.trim();
            const resultsDiv = document.getElementById('weather-results');
            const output = document.getElementById('mock-weather');

            if (!location) {
                alert('Please enter a location!');
                return;
            }

            resultsDiv.classList.add('hidden');
            output.innerHTML = 'Fetching data...';

            setTimeout(() => {
                let report;
                if (location.toLowerCase().includes('chennai') || location.toLowerCase().includes('tamil')) {
                    report = `**Location:** ${location}<br>
                              **Current:** 31°C, Humidity 75%. Mostly Sunny.<br>
                              **7-Day Forecast:** Light showers on Thursday.<br>
                              **Irrigation Advice:** Soil Moisture Low. Recommend **deep watering** today and wait for Thursday's expected rain.`;
                } else {
                    report = `**Location:** ${location}<br>
                              **Current:** 25°C, Humidity 40%. Clear skies.<br>
                              **7-Day Forecast:** Dry and stable.<br>
                              **Irrigation Advice:** Soil Moisture Moderate. Continue current schedule. No immediate action needed.`;
                }
                
                output.innerHTML = report;
                resultsDiv.classList.remove('hidden');
            }, 2000);
        }

        // --- Stage 5: Market Mock Logic ---
        function simulateMarket() {
            const commodity = document.getElementById('marketCommodity').value;
            const resultsDiv = document.getElementById('market-results');
            const output = document.getElementById('mock-market');

            if (!commodity) {
                alert('Please select a commodity!');
                return;
            }

            resultsDiv.classList.add('hidden');
            output.innerHTML = 'Checking market trends...';

            setTimeout(() => {
                let price, trend, advice;

                if (commodity === 'Paddy') {
                    price = '₹2,350 / quintal';
                    trend = '+1.5% (Up)';
                    advice = 'HOLD. Price forecast shows a small rise next week.';
                } else if (commodity === 'Tomato') {
                    price = '₹25 / kg';
                    trend = '-5.2% (Down)';
                    advice = 'SELL TODAY. Prices are declining due to high supply.';
                } else if (commodity === 'Wheat') {
                    price = '₹2,100 / quintal';
                    trend = '0% (Stable)';
                    advice = 'HOLD/SELL AS NEEDED. Stable price, no major forecast changes.';
                } else {
                    price = 'N/A';
                    trend = 'N/A';
                    advice = 'Data not found for this commodity in nearby mandi.';
                }

                output.innerHTML = `**Commodity:** ${commodity}<br>
                                    **Nearest Mandi Price:** ${price}<br>
                                    **24hr Trend:** <span class="font-bold text-lg">${trend}</span><br>
                                    **Sell Recommendation:** <span class="font-bold text-krishi-orange">${advice}</span>`;
                resultsDiv.classList.remove('hidden');
            }, 2000);
        }


        // --- Initial App Load ---
        document.addEventListener('DOMContentLoaded', () => {
            // Start the application on the Home page
            navigateTo('home');
        });
    </script>
</body>
</html>
