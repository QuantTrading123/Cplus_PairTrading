import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
res = np.vstack((arr1, arr2))
print(np.shape(res))
t = np.vstack((np.eye(2), -np.eye(2)))
print(t)
