import numpy as np

a = np.array([[1, 1, 3], [1, 10, 1]])

max_index_col = np.argmax(a, axis=0)
max_index_row = np.argmax(a, axis=1)
print('index row ', max_index_row)
print('index col ', max_index_col)
print('array ', a)


temp = np.transpose(np.asarray(np.where(a == a.max())))
indices = temp[0]
print(indices)
print(indices.shape)



