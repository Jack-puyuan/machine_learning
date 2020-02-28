# TODO: Do your package imports here.
import numpy as np

import matplotlib.pyplot as plt

# TODO: Load the data and the labels from files.
data = np.load('../Iris_data.npy')
datalabel = np.load('../Iris_labels.npy')
# TODO: Plot two feature dimensions against eachother, labeling the axes accordingly.
plt.scatter(data[:, 0], data[:, 1], c=datalabel)

plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Iris Data Set')
# TODO: Make sure the markers in the plot are coloured with their respective class labels.
plt.show()