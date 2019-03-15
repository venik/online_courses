import numpy as np
from setup_problem import load_problem

import matplotlib.pyplot as plt


def squareloss(X, y, w):
    # Average square error
    residuals = X.dot(w) - y
    return np.dot(residuals, residuals) / len(y)


def shooting(X, y, lmbd=1, zero_start=False, random_coordinate=False):
    n, num_bfs = X.shape

    # Zero weights VS Ridge regression weights
    if zero_start:
        w = np.zeros((400,))
    else:
        w = np.linalg.inv(X.T.dot(X) + lmbd * np.eye(num_bfs)).dot(X.T.dot(y))

    old_w = np.copy(w)

    # print('w shape: ' + str(w.shape))

    XX2 = X.T.dot(X) * 2

    coordinates = np.random.permutation(range(len(w))) if random_coordinate else range(len(w))
    for k in range(1000):
        # for j in range(X.shape[1]):
        for j in coordinates:
            aj = XX2[j, j]
            cj = 2 * X[:, [j]].T.dot(y - X.dot(w) + w[j] * X[:, j])
            positive_part = np.sign(cj) * (np.abs(cj) - lmbd) if np.abs(cj) - lmbd > 0 else 0
            # print('\tbefore j: {:} w[j]: {:}'.format(j, w[j]))
            # print('\t => aj: {:} cj: {:} positive: {:} diff: {:}'.format(aj, cj, positive_part, (np.abs(cj) - lmbd)))
            w[j] = 0 if aj == 0 and cj == 0 else positive_part / aj
            # print('\tafter j: {:} w[j]: {:}'.format(j, w[j]))

        # Average square error
        residuals = X.dot(w) - y
        score = np.dot(residuals, residuals) / len(y)
        diff = np.sum(np.abs(old_w - w))
        # print("k: {:} diff: {:.4f} score: {:.4f}".format(k, diff, score))

        if diff < 10e-8:
            print("early exit on k: {:} with score: {:.8f}".format(k, score))
            break
        # update w
        old_w = np.copy(w)

    return w


def main():
    lasso_data_fname = "lasso_data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

    # Generate features
    X_train = featurize(x_train)
    X_val = featurize(x_val)

    print('x_train shape: ' + str(x_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('X_train shape: ' + str(X_train.shape))

    plt.scatter(x_train, y_train, marker='^', c='g')

    best_score = np.finfo(np.float32).max
    best_l2reg = 0.0
    legend = []
    l2reg_costs = []
    l2reg_range = [0.0001, 0.01, .1, .5, 1, 1.5, 1.75, 2, 5, 10, 20]
    x_plot = np.linspace(0, 1, 1000)

    for l2reg in l2reg_range:
        w = shooting(X_train, y_train, lmbd=l2reg, zero_start=False, random_coordinate=False)

        # no regularization
        if l2reg == 0.0001:
            legend.append('Regular regression')
            plt.plot(x_plot, featurize(x_plot).dot(w))

        score = squareloss(X_val, y_val, w)
        l2reg_costs.append(score)

        score_train = squareloss(X_train, y_train, w)
        if best_score > score:
            best_score = score
            best_weights = w
            best_l2reg = l2reg

        print('l2reg: {:.4f} validation score: {:.4f} train score: {:.4f}'.format(l2reg, score, score_train))

    legend.append('Best lasso regression estimation, lambda: {:.4f}'.format(best_l2reg))
    plt.plot(x_plot, featurize(x_plot).dot(best_weights), '-r')

    plt.legend(legend)
    plt.title('Lasso regression')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
