import numpy as np
import matplotlib.pyplot as plt
import skimage.feature
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler # this is for normalising our data
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import tensorflow.keras as keras
import tensorflow as tf

trnImages = np.load('../trnImage.npy')
tstImage = np.load('../tstImage.npy')
trnLabel = np.load('../trnLabel.npy')
tstLabel = np.load('../tstLabel.npy')
trnidx = 20

trnLabel= trnLabel.ravel()
tstLabel= tstLabel.ravel()
print(tstLabel.shape)
number_of_train_img= trnImages.shape[3]
number_of_test_img= tstImage.shape[3]
print(number_of_test_img)
print(number_of_train_img)
