# attempt to do
# https://davidrosenberg.github.io/mlcourse/Archive/2018/Homework/hw2.pdf

import numpy as np

X = np.array((1, 2, 3, 4)).reshape(2, 2)
y = np.array((1, 2)).reshape(-1, 1)
w = np.array((3, 4)).reshape(-1, 1)

print('X: \n' + str(X))
print('y: \n' + str(y))
print('w: \n' + str(w))


XX2 = 2 * X.T.dot(X)
print('XX2:\n' + str(XX2))
print('Xi:\n' + str(X[0, :]))

# Slow and fast implementation of the Shooting algorithm
for j in range(X.shape[1]):
    cj = 0
    aj = 0
    for i in range(X.shape[0]):
        aj += X[i, j]**2
        cj += X[i, j] * (y[i] - w.T.dot(X[i, :]) + w[j] * X[i, [j]])

    aj *= 2
    cj *= 2
    print('aj:' + str(aj))
    print('cj:' + str(cj))

    cj_tmp = 2 * X[:, [j]].T.dot(y - X.dot(w) + w[j] * X[:, [j]])
    print('aj_tmp: ' + str(XX2[j, j]))
    print('cj_tmp:' + str(cj_tmp))
