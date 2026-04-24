import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Activation functions
def sigmoid_act(x, der=False):
    if der:
        return sigmoid_act(x) * (1 - sigmoid_act(x))
    return 1 / (1 + np.exp(-x))

def relu_act(x, der=False):
    if der:
        return np.heaviside(x, 1)
    return np.maximum(x, 0)

def tanh_act(x, der=False):
    if der:
        return 1 - np.tanh(x)**2
    return np.tanh(x)

def identity_act(x, der=False):
    if der:
        return np.ones_like(x)
    return x

# Model implementation
class MLP:
    def __init__(self, hidden_neurons, eta, epochs, input_size, output_size, activation):
        self.hidden_neurons = hidden_neurons
        self.eta = eta
        self.epochs = epochs
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        # Select activation function
        if activation == 'sigmoid':
            self.activation = sigmoid_act
        elif activation == 'relu':
            self.activation = relu_act
        elif activation == 'tanh':
            self.activation = tanh_act
        elif activation == 'identity':
            self.activation = identity_act
        else:
            raise ValueError(f"Activation function '{activation}' not supported.")

        # Initialize weights and biases
        np.random.seed(42)
        self.w1 = np.random.rand(self.input_size, self.hidden_neurons)
        self.b1 = np.zeros((1, self.hidden_neurons))
        self.wOut = np.random.rand(self.hidden_neurons, self.output_size)
        self.bOut = np.zeros((1, self.output_size))

        # Store losses
        self.train_losses = []
        self.val_losses = []

    def feedforward(self, x):
        z1 = self.activation(np.dot(x, self.w1) + self.b1)
        y = sigmoid_act(np.dot(z1, self.wOut) + self.bOut)
        return z1, y

    def calculate_loss(self, y, target):
        loss = -target * np.log(y) - (1 - target) * np.log(1 - y)
        return np.mean(loss)

    def backpropagation(self, x, z1, y, target):
        loss = self.calculate_loss(y, target)

        # Delta
        delta_out = (y - target)
        delta_hidden = np.dot(delta_out, self.wOut.T) * self.activation(z1, der=True)

        # Update weights and biases
        self.wOut -= self.eta * np.dot(z1.T, delta_out)
        self.bOut -= self.eta * np.sum(delta_out, axis=0, keepdims=True)
        self.w1 -= self.eta * np.dot(x.reshape(1, -1).T, delta_hidden)
        self.b1 -= self.eta * np.sum(delta_hidden, axis=0, keepdims=True)

        return loss

    def fit(self, X_train, y_train, X_val, y_val):
        for epoch in range(self.epochs):
            epoch_loss = 0

            # Training phase
            for x, target in zip(X_train, y_train):
                # Feedforward
                z1, y = self.feedforward(x)

                # Backpropagation and update weights
                loss = self.backpropagation(x, z1, y, target)
                epoch_loss += loss

            # Calculate average epoch loss
            epoch_loss /= len(X_train)
            self.train_losses.append(epoch_loss)

            # Validation phase
            val_loss = 0
            for x, target in zip(X_val, y_val):
                _, y = self.feedforward(x)
                val_loss += self.calculate_loss(y, target)

            # Calculate validation loss
            val_loss /= len(X_val)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{self.epochs}, Train Loss: {epoch_loss}, Val Loss: {val_loss}")

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            _, y = self.feedforward(x)
            predictions.append(np.round(y)[0])
        return np.array(predictions)


df = pd.read_csv('diabetes.csv')

# Standardize the features
X = df.iloc[:, :-1]
y = df['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Train the model
mlp = MLP(hidden_neurons=4, eta=0.0002, epochs=100, input_size=X_train.shape[1], output_size=1, activation='identity')
mlp.fit(X_train, y_train, X_val, y_val)

# Plot loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(mlp.train_losses, label='Training Loss')
plt.plot(mlp.val_losses, label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# Evaluation
predictions = mlp.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
