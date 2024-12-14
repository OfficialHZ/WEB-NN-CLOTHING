// Global variable to hold the TensorFlow model
let model;

// Load the TensorFlow model
async function loadModel() {
    console.log("Loading model...");
    try {
        model = await tf.loadLayersModel('./modeloexportado/model.json'); // Load pre-trained model
        console.log("Model loaded successfully.");
        model.summary(); // Display model details in the console
        classifyButton.disabled = false; // Enable classify button
    } catch (error) {
        console.error("Error loading model:", error); // Log any errors
    }
}

loadModel(); // Call the function to load the model

// DOM elements
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true }); // Allow frequent reads for better performance
const fileInput = document.getElementById('file-input');
const classifyButton = document.getElementById('classify-button');
const clearButton = document.getElementById('clear-button');
const resultDisplay = document.getElementById('prediction-result');

// Disable classify button initially (until model is loaded)
classifyButton.disabled = true;

// Handle image upload
fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Clear canvas and set a white background
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#ffffff"; // Set white background
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Draw the uploaded image onto the canvas
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = e.target.result; // Set image source from file reader
        };
        reader.readAsDataURL(file); // Read file as data URL
    } else {
        console.error("No file selected."); // Log if no file is selected
    }
});

// Handle classify button click
classifyButton.addEventListener('click', async () => {
    try {
        // Extract image data from canvas
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Convert image data to TensorFlow tensor
        const input = tf.browser.fromPixels(imageData, 1) // Convert to grayscale
            .resizeBilinear([28, 28])                    // Resize to 28x28 pixels
            .toFloat()                                   // Convert to float values
            .div(255)                                    // Normalize to range [0, 1]
            .expandDims(0);                              // Add batch dimension

        // Debugging: Log preprocessed input
        console.log("Preprocessed input tensor:", input.dataSync().slice(0, 10));

        // Check for empty input
        if (input.dataSync().every(value => value === 0)) {
            resultDisplay.innerText = "Please provide a valid image.";
            return;
        }

        // Perform prediction
        const prediction = await model.predict(input).array();
        console.log("Prediction probabilities:", prediction[0]);

        // Map output to class names
        const classNames = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
        ];
        const predictedClassIndex = prediction[0].indexOf(Math.max(...prediction[0])); // Find index of max probability
        const predictedClass = classNames[predictedClassIndex]; // Get class name
        resultDisplay.innerText = `Prediction: ${predictedClass}`;
    } catch (error) {
        console.error("Error during classification:", error); // Log errors
        resultDisplay.innerText = "Error during prediction.";
    }
});

// Handle clear button click
clearButton.addEventListener('click', () => {
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff"; // Reset to white background
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Reset result display
    resultDisplay.innerText = "Results will appear here";
    fileInput.value = ""; // Reset file input
});
