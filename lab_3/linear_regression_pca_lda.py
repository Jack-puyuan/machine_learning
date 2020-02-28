# TODO: Do your package imports here.
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Use sklearn for Linear Regression
# TODO: Load the regression X and Y values as in Task 3.1.
# TODO: Instantiate a LinearRegression model object.
# TODO: Fit the model to the observations.
# TODO: Print the intercept and coefficient of the model.
# TODO: Predict the Y values of X so that we can observe our linear model.
# TODO: Scatter plot the data and then plot the predicted Y values on top.
# TODO: Give figure suitable axes labels and title.
x = np.load('../x_points.npy').reshape(1, 10)
y = np.load('../y_observations.npy').reshape(1, 10)

print(x)
print(x.shape)
print('----------------------------------abs---')
model = LinearRegression()
model.fit(x, y)
pred_y = model.predict(x)
intercept = model.intercept_
print(intercept)
coefficient = model.coef_
print(coefficient)
plt.figure()
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x, y, c='red')
plt.plot(x, pred_y, c='blue')
plt.show()

# Use sklearn for PCA
# TODO: Load the wine dataset as in Task 3.2.
# TODO: Instantiate a PCA model object with a number of components as chosen in Task 3.2.
# TODO: Fit the model to the data and project it to the new space.
# TODO: Plot the projected data points.
# TODO: Give figure suitable axes labels and title.
x1 = np.load('../wineData.npy')
pca = PCA(n_components=2)
new_data = pca.fit_transform(x1)
print(new_data.shape)

plt.figure()
plt.title('2 dimensions of principal components')
plt.xlabel('1 dimension')
plt.ylabel('2 dimension')
plt.scatter(new_data[:, 0], new_data[:, 1])
plt.show()

# Use sklearn for LDA
# TODO: Load the wine dataset as in Task 3.3.
# TODO: Instantiate a LinearDiscriminantAnalysis model object with a number of components as chosen in Task 3.3.
# TODO: Fit the model to the data and project it to the new space.
# TODO: Plot the projected data points.
# TODO: Give figure suitable axes labels and title.
x2 = np.load('../wineData.npy')
y2 = np.load('../winelabels.npy')
lda = LinearDiscriminantAnalysis(n_components=2, priors=None, shrinkage=None,
                                 solver='svd', store_covariance=False, tol=0.0001)
lda.fit(x2, y2)
new_labels = lda.predict(x2)
plt.figure()
plt.title('linear discrimiant analysis')
plt.xlabel('linear discrimiant 1')
plt.ylabel('linear discrimiant 2')
plt.scatter(x2[:, 0], x2[:, 1], c=new_labels)
plt.show()
