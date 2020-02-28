import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler  # this is for normalising our data


# Define a function which takes in our dataset and labels, and returns a train/test split of the data
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

# Create a binary class problem.
positive_class = 0  # This means we are interested in classifying class 0 vs. all other classes. Try changing this to another class if you want.

temp_class = np.max(y) + 1
y[y != positive_class] = temp_class
y[y != temp_class] = 0
y[y == temp_class] = 1

# Create a 2D dataset. This is just to aide in visualisation steps later on.
x = x[:, [0, 1]]

# Plot the data, colouring the samples by their binary label.
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Observed sample points for Binary SVM')
plt.show()

# Split the dataset into a training and testing set.
train_data, train_labels, test_data, test_labels = train_test_split(x, y, ratio=0.2)

# Normalise the data based on the training set
normaliser = StandardScaler().fit(train_data)  # fit the data normalizer to the observed training samples
train_data = normaliser.transform(train_data)
test_data = normaliser.transform(test_data)

# Plot the training samples, then plot the testing samples with different marker shapes.
plt.scatter(train_data[:, 0], train_data[:, 1], marker='o', c=train_labels)
plt.scatter(test_data[:, 0], test_data[:, 1], marker='x', c=test_labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Train/Test split of feature space')
plt.show()

C = 1  # This is the penalty of the error term (Lecture 8, slide 18).
kernel_type = 'linear'  # This denotes the type of kernel used by our SVM (Lecture 8, slides 15/16)
model = SVC(kernel=kernel_type, C=C)

# Fit the model to the dataset, given the binary labels.
model.fit(train_data, train_labels)

# Use the model to predict the class of the test observations.
predicted_labels = model.predict(test_data)

# Plot the training and testing data.
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_labels, marker='o')
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_labels, marker='x')

# Get axis limits and create a grid to evaluate the model upon.
ax = plt.gca()
xlimits = ax.get_xlim()
ylimits = ax.get_ylim()
xx = np.linspace(xlimits[0], xlimits[1], 90)
yy = np.linspace(ylimits[0], ylimits[1], 90)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Plot decision boundary and margins.
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Highlight the support vectors.
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.show()

# Calculate the percentage accuracy rate on test set
accuracy = np.sum(np.equal(test_labels, predicted_labels)) / test_labels.shape[0] * 100
print('Percentage accuracy on testing set is: {0:.2f}%'.format(accuracy))