import numpy as np


# a = np.zeros([512,241])
# print(a.shape)
# b = np.zeros([a.shape[0],2])
# print(b.shape)

# a = np.append(a,b, axis=1)
# print(a.shape)

a = np.zeros(241)
print(a.shape)
b = np.zeros(2)
print(b.shape)

a = np.append(a,b,axis=0)
print(a.shape)