# Neural Network from scratch
Comparison of baseline, perceptron, and multilayer perceptron (MLP) models implemented from scratch using Python.


# Project Overview
This project experiments and compares the performance of different model -- logistic regression, perceptron and MLP

The goal is to understand how neural network complexity and hyperparameter tuning affect classification accuracy and learning capability.


## Features

- Perceptron implemented from scratch
- MLP implemented from scratch
- Forward propagation
- Backpropagation
- Gradient descent training
- Model comparison and evaluation
- Accuracy and loss visualization


## Project Structure

Neural Network/
└── Baseline.py         # train and evaluate a logistic regression model as baseline model
└── mlp2.py             # train and evaluate a MLP model
└── Neural Network.py   # train and evaluate logistic regression and perceptron
└── diabetes.csv        # dataset used for model training

## Methodology

### Baseline
A simple reference model used for comparison.

### Perceptron
A single-layer neural network for linear classification.

### MLP
A multilayer neural network with hidden layers capable of learning nonlinear patterns using backpropagation.

## Results

| Model                | Accuracy |
|---------------------|-----------|
| Logistic Regression | 0.75      |
| Perceptron          | 0.72      |
| MLP                 | 0.79      |


## Key Findings
- Logistic regression slightly outperformed the perceptron
- MLP achieves the best accuracy due to the hidden neurons that are able to identify the relationship between attributes
- Increasing one hyperparameter (hidden neurons/epochs) alone does not guarantee better performance


## Technologies used
- Scikit-learn
- Matplotlib
- numpy
- pandas
