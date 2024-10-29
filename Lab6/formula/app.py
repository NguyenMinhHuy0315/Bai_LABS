from flask import Flask, render_template
import torch

app = Flask(__name__)

# Activation Functions
def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def relu(x):
    return torch.max(torch.tensor(0.0), x)

def softmax(zi):
    exp_zi = torch.exp(zi)
    return exp_zi / torch.sum(exp_zi)

def tanh(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

# Loss Functions
def crossEntropyLoss(output, target):
    return -torch.sum(target * torch.log(torch.softmax(output, dim=0)))

def meanSquareError(output, target):
    return torch.sum((output - target) ** 2)

def binaryEntropyLoss(output, target, n):
    return -torch.sum(target * torch.log(output) + (1 - target) * torch.log(1 - output)) / n

@app.route('/')
def index():
    # Sample inputs and target tensors
    inputs = torch.tensor([0.1, 0.3, 0.6, 0.7])
    target = torch.tensor([0.31, 0.32, 0.8, 0.2])
    n = len(inputs)
    
    # Compute losses
    mse = meanSquareError(inputs, target).item()
    binary_loss = binaryEntropyLoss(inputs, target, n).item()
    cross_loss = crossEntropyLoss(inputs, target).item()

    # Compute activations
    x = torch.tensor([1, 5, -4, 3, -2], dtype=torch.float32)
    f_sigmoid = sigmoid(x).tolist()
    f_relu = relu(x).tolist()
    f_softmax = softmax(x).tolist()
    f_tanh = tanh(x).tolist()

    return render_template('index.html', 
                           mse=mse, 
                           binary_loss=binary_loss, 
                           cross_loss=cross_loss, 
                           f_sigmoid=f_sigmoid, 
                           f_relu=f_relu, 
                           f_softmax=f_softmax, 
                           f_tanh=f_tanh)

if __name__ == '__main__':
    app.run(debug=True)
