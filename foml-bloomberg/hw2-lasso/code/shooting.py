# attempt to do
# https://davidrosenberg.github.io/mlcourse/Archive/2018/Homework/hw2.pdf

# Reference: Machine Learning: A Probabilistic Perspective 1st Edition by Kevin P. Murphy. Chapter 13.4.1
# Example from the book
# https://github.com/probml/pmtksupport/blob/master/markSchmidt-9march2011/markSchmidt/lasso/LassoShooting.m

import numpy as np

X = np.array((1, 2, 3, 4)).reshape(2, 2)
y = np.array((1, 2)).reshape(-1, 1)
w = np.array((3, 4)).reshape(-1, 1)
lmbd = 1
vectorized = True

print('X: \n' + str(X))
print('y: \n' + str(y))
print('w: \n' + str(w))
print('lambda: {:} vectorized: {:}'.format(lmbd, vectorized))

XX2 = 2 * X.T.dot(X)
# print('XX2:\n' + str(XX2))
# print('Xi:\n' + str(X[0, :]))

# Slow and fast implementation of the Shooting algorithm
for j in range(X.shape[1]):
    cj = 0
    aj = 0
    for i in range(X.shape[0]):
        aj += X[i, j]**2
        cj += X[i, j] * (y[i] - w.T.dot(X[i, :]) + w[j] * X[i, [j]])

    aj *= 2
    cj *= 2

    aj_vec = XX2[j, j] if vectorized else aj
    cj_vec = 2 * X[:, [j]].T.dot(y - X.dot(w) + w[j] * X[:, [j]]) if vectorized else cj
    print('j: {:} aj_vec: {:}'.format(j, aj_vec))
    print('j: {:} cj_vec: {:}'.format(j, cj_vec))

    positive_part = np.sign(cj_vec) * (np.abs(cj_vec) - lmbd) if np.abs(cj_vec) - lmbd > 0 else 0
    w[j] = 0 if aj_vec == 0 and cj_vec == 0 else positive_part / aj_vec

    print('j: {:} w[j]: {:}'.format(j, w[j]))
