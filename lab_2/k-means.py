import sklearn.cluster
import sklearn.mixture
from sklearn.cluster import KMeans
import numpy as np

import matplotlib.pyplot as plt
data = np.load('../Iris_data.npy')
# datalabel = np.load('../Iris_labels.npy')

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
predicted_clusters = kmeans.predict(data)
centers = kmeans.cluster_centers_
plt.scatter(data[:, 0], data[:, 2], c=predicted_clusters)
plt.scatter(centers[:, 0], centers[:, 2], color='orange', s=120, marker='x')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.title('sklearn implementation of k-means')
plt.show()
