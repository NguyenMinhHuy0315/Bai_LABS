# app.py
import io
import torch
from flask import Flask, request, jsonify, render_template
from torchvision import datasets, transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim

# Initialize the Flask app
app = Flask(__name__)

# Define the model structure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLPModel(nn.Module):
    def __init__(self, n_features):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load CIFAR10 dataset for training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Initialize model, optimizer, and loss function
model = MLPModel(32 * 32 * 3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the model
n_epochs = 2  # For demo purposes, keep epochs small
for epoch in range(n_epochs):
    model.train()
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

# Define a function to preprocess the image and make a prediction
def predict_image(image):
    # Apply transformations
    image = transform(image).unsqueeze(0).to(device)
    # Get model predictions
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Route to upload an image and get a prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Load the image
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    # Get the prediction
    prediction = predict_image(image)
    # Map to class names for CIFAR10
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    return jsonify({'class_id': prediction, 'class_name': classes[prediction]})

# Route to display the HTML form for uploading images
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
