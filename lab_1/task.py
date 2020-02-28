"""
name puyuan yang
date 07/10/2019
student number 994752
"""

import numpy as np

import matplotlib.pyplot as plt

# ------------------------- task 1.1--------------------------------
x = 345
iris_data = 2 * x + 5
print(iris_data)
print("Hello World")


def f(w, x, b):
    return w * x + b


f(2, 345, 5)

# ------------------------- task 1.2--------------------------------
# 1
B = [1, 2, 3, 4, 5, 6, 7, 8]
B.reverse()
for b in B:
    print(b)


# 2
def equals100(x):
    if x == 100:
        return True
    else:
        return False


print(equals100(100))

# 3
matrix = np.random.randint(0, 5, (20, 3), int)
vector = np.random.randint(0, 3, (20, 1), int)
myDictionary = {
    'data_name': 'myData',
    'data': matrix,
    'labels': vector
}
# 4
print(myDictionary)

# ------------------------- task 1.3--------------------------------
# 2
matrix1 = np.random.randint(1, 8, (2, 3), int)
matrix2 = np.random.randint(1, 8, (3, 4), int)

# 3
multiplyResult = np.dot(matrix1, matrix2)

print(multiplyResult)
# 4
print(multiplyResult[:, 0])

# ------------------------- task 1.4--------------------------------

# 1
iris_data = np.load('../Iris_data.npy')
# 2
print(iris_data.shape)
# 3
plt.scatter(iris_data[:, 0], iris_data[:, 1])
# 4
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Iris Data Set')
plt.show()
# ------------------------- task 1.5--------------------------------

