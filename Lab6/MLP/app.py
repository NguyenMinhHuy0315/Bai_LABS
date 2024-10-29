import torch
import torch.nn as nn
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

# Load the saved MLP model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("C:/Users/Admin/Documents/VS/HMvUD/Lab6/MLP/MLP_dress.pth", map_location=device)
model.eval()

# Define the transformation to match the training transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # Convert to grayscale
    transforms.Resize((28, 28)),                 # Resize to 28x28 pixels
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Preprocess the image for the model
    image = Image.open(file).convert('L')  # Convert to grayscale if not already
    image = transform(image).unsqueeze(0).to(device)  # Apply transforms and add batch dimension

    # Get prediction from the model
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = predicted.item()

    return render_template('index.html', prediction=label)

if __name__ == "__main__":
    app.run(debug=True)
