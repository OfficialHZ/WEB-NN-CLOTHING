let model;

// Load the model
async function loadModel() {
    console.log("Loading model...");
    try {
        model = await tf.loadLayersModel('./modeloexportado/model.json');
        console.log("Model loaded successfully.");
        model.summary();
        classifyButton.disabled = false; // Enable button after model loads
    } catch (error) {
        console.error("Error loading model:", error);
    }
}

loadModel();

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true }); // Improve performance
const fileInput = document.getElementById('file-input');
const classifyButton = document.getElementById('classify-button');
const clearButton = document.getElementById('clear-button');
const resultDisplay = document.getElementById('prediction-result');

// Disable classify button initially
classifyButton.disabled = true;

fileInput.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                // Clear the canvas and set a white background
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = "#ffffff"; // Set background to white
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Draw the image onto the canvas
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    } else {
        console.error("No file selected.");
    }
});

classifyButton.addEventListener('click', async () => {
    try {
        // Get the image data from the canvas
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        // Debug: Log raw pixel data
        console.log("Raw canvas pixel data (first 10 values):", imageData.data.slice(0, 10));

        // Convert canvas data to a TensorFlow tensor
        const input = tf.browser.fromPixels(imageData, 1) // Grayscale (1 channel)
            .resizeBilinear([28, 28])                    // Resize to 28x28
            .toFloat()                                  // Convert to float
            .div(255)                                   // Normalize to [0, 1]
            .expandDims(0);                             // Add batch dimension

        // Debugging: Log preprocessed tensor values
        const inputValues = input.dataSync();
        console.log("Preprocessed tensor (first 10 values):", inputValues.slice(0, 10));

        // Check for invalid or empty input
        if (inputValues.every(value => value === 0)) {
            console.warn("Input is empty or invalid.");
            resultDisplay.innerText = "Please provide a valid image.";
            return;
        }

        // Predict the class probabilities
        const prediction = await model.predict(input).array();
        console.log("Prediction probabilities:", prediction[0]);

        // Class names corresponding to the model's output
        const classNames = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
        ];

        // Find the predicted class
        const predictedClassIndex = prediction[0].indexOf(Math.max(...prediction[0]));
        const predictedClass = classNames[predictedClassIndex];
        resultDisplay.innerText = `Prediction: ${predictedClass}`;
    } catch (error) {
        console.error("Error during classification:", error);
        resultDisplay.innerText = "Error during prediction. Check console for details.";
    }
});

// Clear canvas and reset prediction display
clearButton.addEventListener('click', () => {
    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Reset canvas background to white
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Reset the prediction display
    resultDisplay.innerText = "Prediction will appear here";

    // Reset file input
    fileInput.value = "";
});
