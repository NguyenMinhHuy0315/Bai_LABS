from flask import Flask, render_template
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Load Wine dataset
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Split the dataset into training and testing sets (70:30)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and fit the KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Predict on the test set
    y_pred = knn.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Pass metrics to the HTML template
    return render_template('index.html', accuracy=accuracy, recall=recall, precision=precision, cm=cm)

if __name__ == '__main__':
    app.run(debug=True)
