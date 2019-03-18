import numpy as np
from setup_problem import load_problem
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt

# Reference: Machine Learning: A Probabilistic Perspective 1st Edition by Kevin P. Murphy. Chapter 13.4.1
# Example from the book
# https://github.com/probml/pmtksupport/blob/master/markSchmidt-9march2011/markSchmidt/lasso/LassoShooting.m


class LassoRegression(BaseEstimator, RegressorMixin):
    def __init__(self, l1reg=1, zero_start=False, random_coordinate=False):
        if l1reg <= 0.0:
            raise ValueError('penalty cannot be negative or zero')
        self.l1reg = l1reg
        self.zero_start_ = zero_start
        self.random_coordinate_ = random_coordinate

    def fit(self, X, y):
        n, num_bfs = X.shape

        def shooting(X, y):
            old_w = np.copy(w)
            XX2 = X.T.dot(X) * 2

            coordinates = np.random.permutation(range(len(w))) if self.random_coordinate_ else range(len(w))
            for k in range(1000):
                for j in coordinates:
                    aj = XX2[j, j]
                    cj = 2 * X[:, [j]].T.dot(y - X.dot(w) + w[j] * X[:, j])
                    positive_part = np.sign(cj) * (np.abs(cj) - self.l1reg) if np.abs(cj) - self.l1reg > 0 else 0
                    w[j] = 0 if aj == 0 and cj == 0 else positive_part / aj

                # Average square error
                residuals = X.dot(w) - y
                score = np.dot(residuals, residuals) / len(y)
                diff = np.sum(np.abs(old_w - w))

                if diff < 10e-8:
                    print("early exit on k: {:} with score: {:.8f}".format(k, score))
                    break
                # update w
                old_w = np.copy(w)

            return w

        # Zero weights VS Ridge regression weights
        if self.zero_start_:
            w = np.zeros((num_bfs,))
        else:
            w = np.linalg.inv(X.T.dot(X) + self.l1reg * np.eye(num_bfs)).dot(X.T.dot(y))

        self.w_ = shooting(X, y)
        return self

    def predict(self, X):
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        # print(str(self.w_))
        return np.dot(X, self.w_)

    def score(self, X, y):
        # Average square error
        try:
            getattr(self, "w_")
        except AttributeError:
            raise RuntimeError("You must train classifer before predicting data!")
        residuals = self.predict(X) - y
        return np.dot(residuals, residuals)/len(y)


def main():
    lasso_data_fname = "lasso_data.pickle"
    x_train, y_train, x_val, y_val, target_fn, coefs_true, featurize = load_problem(lasso_data_fname)

    # Generate features
    X_train = featurize(x_train)
    X_val = featurize(x_val)

    print('x_train shape: ' + str(x_train.shape))
    print('y_train shape: ' + str(y_train.shape))
    print('X_train shape: ' + str(X_train.shape))

    lmbd_max = np.max(2 * X_train.T.dot(y_train))
    print('Max lambda: {:.4f}'.format(lmbd_max))

    best_score = np.finfo(np.float32).max
    best_estimator = None
    best_l1reg = 0.0
    legend = []
    l1reg_costs = []
    l1reg_train_costs = []
    # l1reg_range = [0.0001, 0.01, .1, .5, 1, 1.5, 1.75, 2, 5, 10, 20]
    l1reg_range = lmbd_max * np.power(.8, range(30))
    x_plot = np.linspace(0, 1, 1000)

    f1=plt.figure(1)
    plt.scatter(x_train, y_train, marker='^', c='g')

    for l1reg in l1reg_range:
        lasso = LassoRegression(l1reg=l1reg, zero_start=False, random_coordinate=False)
        lasso.fit(X_train, y_train)

        # no regularization
        if l1reg == 0.0001:
            legend.append('Regular regression')
            plt.plot(x_plot, lasso.predict(featurize(x_plot)))

        score = lasso.score(X_val, y_val)
        l1reg_costs.append(score)

        score_train = lasso.score(X_train, y_train)
        l1reg_train_costs.append(score_train)

        if best_score > score:
            best_score = score
            best_estimator = lasso
            best_l1reg = l1reg

        print('l1reg: {:.4f} validation score: {:.4f} train score: {:.4f}'.format(l1reg, score, score_train))

    legend.append('Best lasso regression estimation, lambda: {:.4f}'.format(best_l1reg))
    plt.plot(x_plot, best_estimator.predict(featurize(x_plot)), '-r')

    print('length of vector w: {:} number of non-zero elements: {:}'.format(len(best_estimator.w_), np.count_nonzero(best_estimator.w_)))

    plt.legend(legend)
    plt.title('Lasso regression')
    plt.grid()
    f1.show()

    f2=plt.figure(2)
    plt.plot(l1reg_range, np.log(l1reg_costs), '-r^', l1reg_range, np.log(l1reg_train_costs), '-g*')
    plt.legend(['Test set score', 'Train set score'])
    plt.grid()

    f2.show()
    plt.show()


if __name__ == '__main__':
    main()
