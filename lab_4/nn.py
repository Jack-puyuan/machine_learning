import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler # this is for normalising our data
print(tf.__version__)

def train_test_split(data, labels, ratio=0.2):
    np.random.seed(5)  # set random generator seed for reproducibility

    # Get sample indices for each set
    testing_indices = np.random.choice(np.arange(data.shape[0]), int(data.shape[0] * ratio), replace=False)
    training_indices = np.delete(np.arange(data.shape[0]), testing_indices)

    # Divide the dataset into the two sets.
    test_data = data[testing_indices, :]
    test_labels = labels[testing_indices]
    train_data = data[training_indices, :]
    train_labels = labels[training_indices]

    return train_data, train_labels, test_data, test_labels

# Load the original Iris data and labels
x = np.load('../Iris_data.npy')
y = np.load('../Iris_labels.npy')

# Make 2D to conform with the above experiments
x = x[:, [0, 1]]

# Find out the number of classes in the dataset
number_of_classes = np.max(y) + 1

# Select data points to divide into a training and testing set
train_data, train_labels, test_data, test_labels = train_test_split(x, y, ratio=0.2)

# Normalise the data based on the training set
normaliser = StandardScaler().fit(train_data)
train_data = normaliser.transform(train_data)
test_data = normaliser.transform(test_data)
