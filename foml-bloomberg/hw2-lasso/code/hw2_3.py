import numpy as np
from setup_problem import load_problem

def main():
    lasso_data_fname = "lasso_data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

    # Generate features
    X_train = featurize(x_train)
    X_val = featurize(x_val)

    n, num_bfs = X_train.shape
    lmbd = 1

    w = np.linalg.inv(X_train.T.dot(X_train) + lmbd * np.eye(num_bfs)).dot(X_train.T.dot(y_train))

    print('x_train shape: ' + str(x_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('X_train shape: ' + str(X_train.shape))
    print('w shape: ' + str(w.shape))

    XX2 = X_train.T.dot(X_train) * 2

    for j in range(X_train.shape[1]):
        cj = 0
        aj = 0
        for i in range(X_train.shape[0]):
            aj += X_train[i, j] ** 2
            cj += X_train[i, j] * (y_train[i] - w.T.dot(X_train[i, :]) + w[j] * X_train[i, [j]])

        aj *= 2
        cj *= 2
        print('aj: ' + str(XX2[j, j]))
        print('cj:' + str(cj))

        aj_tmp = XX2[j, j]
        cj_tmp = 2 * X_train[:, [j]].T.dot(y_train - X_train.dot(w) + w[j] * X_train[:, j])
        print('aj_tmp: ' + str(aj_tmp))
        print('cj_tmp:' + str(cj_tmp))

if __name__ == '__main__':
    main()
