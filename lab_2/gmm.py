import sklearn.cluster
import sklearn.mixture
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

data = np.load('../Iris_data.npy')

gmm = sklearn.mixture.GaussianMixture(n_components=3)
gmm.fit(data)
labels = gmm.predict(data)
plt.scatter(data[:, 0], data[:, 2], c=labels)
pcentroids = gmm.means_
plt.scatter(pcentroids[:, 0], pcentroids[:, 2], color='orange', s=120, marker='x')
plt.xlabel('sepal length')
plt.ylabel('petal length')

plt.title('sklearn implementation of GMMs')
plt.show()
