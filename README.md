## Artificial Neural Network for Fish Species Classification

This project demonstrates how to use an Artificial Neural Network (ANN) to classify different fish species based on given features. We utilize a dataset that contains several features of fish and train a neural network to predict the species accurately.

## Project Overview

The goal of this project is to classify fish species based on their characteristics using an Artificial Neural Network (ANN). We utilize the dataset provided in the project and preprocess the data before feeding it into a neural network model for training and evaluation.

## Dataset

You can find the dataset in the Kaggle.
https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

## Project Structure

ann-fish.ipynb: The Jupyter Notebook that contains the full implementation of the ANN model, from data preprocessing to model evaluation.
README.md: Project overview, instructions, and explanations.

## Requirements

To run this project, you will need the following Python libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import random
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Sequential

To run the code, install the required libraries.

## Project Steps

1. Importing Essential Libraries
We import the essential libraries to practice this task. Such libraries are numpy, tensorflow, pandas, matplotlib, seaborn, sklearn etc.
2. Data Loading and Preprocessing
We load the dataset and perform exploratory data analysis to understand the distribution and relationship of the features.
After understanding the distribution, we normalize and flatten our image values to provide our model a stable data.
3. Building the ANN Model
The neural network is built using TensorFlow. It consists of:
Input layer: Accepting the fish feature set.
Three hidden layers with activation functions (ReLU).
Output layer: Using softmax to classify the fish species.
4. Model Training
The model is trained using the categorical cross-entropy loss function, Adam optimizer, and accuracy as the evaluation metric.
We split the dataset into training and testing sets to evaluate the model’s performance.
5. Model Examining
We examine our model’s loss and accuracy graphs to see if there are any overfit signs.
6. Model Evaluation
After training, the model's accuracy and loss values are evaluated on the test dataset.
Visualizations are provided to show the model's performance, accuracy trends, and confusion matrix for a deeper insight into the results.
7. Conclusion
The trained ANN model effectively classifies the fish species, and the evaluation metrics show the accuracy achieved. Further fine-tuning of the model and feature engineering can be applied to improve the results.

## Running the Project

Clone the repository.
Install the required libraries.
Open the ann-fish.ipynb notebook and run the cells to train and evaluate the ANN model.

## Results

The model achieves an accuracy of around %97 on the test data. Below are some of the results visualized during the analysis:
Accuracy and loss plots during training.
Confusion matrix showing the classification results for the fish species.

## Future Improvements

Hyperparameter tuning: Exploring different learning rates, optimizers, and batch sizes to improve performance.
Data augmentation: Implementing techniques to augment the dataset for better generalization.

## Acknowledgments

Special thanks to Kaggle for hosting the dataset and providing a platform to run this project.
