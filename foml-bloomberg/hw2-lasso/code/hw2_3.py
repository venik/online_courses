import numpy as np
from setup_problem import load_problem

def main():
    lasso_data_fname = "lasso_data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

    # Generate features
    X_train = featurize(x_train)
    X_val = featurize(x_val)

    n, num_bfs = X_train.shape
    lmbd = 0.00001

    w = np.linalg.inv(X_train.T.dot(X_train) + lmbd * np.eye(num_bfs)).dot(X_train.T.dot(y_train))
    old_w = np.copy(w)

    print('x_train shape: ' + str(x_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('X_train shape: ' + str(X_train.shape))
    print('w shape: ' + str(w.shape))

    lmbd = 1
    XX2 = X_train.T.dot(X_train) * 2

    for k in range(10):
        for j in range(X_train.shape[1]):
            aj = XX2[j, j]
            cj = 2 * X_train[:, [j]].T.dot(y_train - X_train.dot(w) + w[j] * X_train[:, j])
            positive_part = np.sign(cj) * (np.abs(cj) - lmbd) if np.abs(cj) - lmbd > 0 else 0
            # print('\tbefore j: {:} w[j]: {:}'.format(j, w[j]))
            # print('\t => aj: {:} cj: {:} positive: {:} diff: {:}'.format(aj, cj, positive_part, (np.abs(cj) - lmbd)))
            w[j] = 0 if aj == 0 and cj == 0 else positive_part / aj
            # print('\tafter j: {:} w[j]: {:}'.format(j, w[j]))

        diff = np.sum(np.abs(old_w - w))
        print("k: {:} diff: {:.2f}".format(k, diff))

        # update w
        old_w = np.copy(w)

if __name__ == '__main__':
    main()
