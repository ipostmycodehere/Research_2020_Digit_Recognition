import numpy as np
from numpy import linalg as LA

def pca(X, k):
    #Move data to the origin
    mean = np.mean(X, axis=0)
    data = X - mean

    #Compute covariance matrix
    S = 1/data.shape[0]*X.transpose().dot(X)

    #Eigendecomposition
    num_iter = 100
    eigen_vectors = np.zeros((k,X.shape[1]))
    eigen_values = np.zeros(k)
    for i in range(k):
        v = np.random.rand(X.shape[1],1)
        for j in range(num_iter):
            v = S.dot(v)/LA.norm(S.dot(v),2)
        eigen_vectors[i,:] = v.transpose()
        eigen_values[i] = v.transpose().dot(S).dot(v)
        S = S - eigen_values[i]*v.dot(v.transpose())

    #Project data
    proj_data = data.dot(eigen_vectors.transpose())
    return [eigen_values,eigen_vectors,proj_data]
