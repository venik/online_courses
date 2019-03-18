import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin

from setup_problem import load_problem


class MyRidge(BaseEstimator, RegressorMixin):
    def __init__(self, l2reg=1):
        if (l2reg < 0):
            raise ValueError('penalty cannot be negative')
        self.l2reg = l2reg

    def fit(self, X, y):
        n, num_bfs = X.shape
        y = y.reshape(-1)

        def objective_func(w):
            pred = np.dot(X, w)
            se = np.sum((pred - y)**2) / n
            objective = se + self.l2reg * np.sum(w**2)
            return objective

        w0 = np.zeros(num_bfs)
        self.w_ = minimize(objective_func, w0).x
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

    # print('Original: ' + str(x_train[0]) + ' Featurized: ' + str(X_train[0]))
    print('Featurized shape: ' + str(X_train[0].shape))

    # Visualize data
    f1 = plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.subplots_adjust(hspace=0.5)

    legend = []
    l2reg_range = np.concatenate((
        np.linspace(0, 1, 8, endpoint=False),
        np.linspace(1, 5, 4)
    ))

    best_score = np.finfo(np.float32).max
    best_estimator = None
    base_weights = np.zeros(X_train[0].shape[0])

    best_l2reg = 0
    l2reg_costs = []
    x_plot = np.linspace(0, 1, 1000)
    for l2reg in l2reg_range:
        ridge_regression_estimator = MyRidge(l2reg=l2reg)
        ridge_regression_estimator.fit(X_train, y_train)

        if l2reg == 0:
            y_plot = ridge_regression_estimator.predict(featurize(x_plot))
            base_weights = ridge_regression_estimator.w_
            plt.plot(x_plot, y_plot)
            legend.append('l2reg {:.4}'.format(l2reg))

        score = ridge_regression_estimator.score(X_val, y_val)
        l2reg_costs.append(score)

        score_train = ridge_regression_estimator.score(X_train, y_train)
        if best_score > score:
            best_score = score
            best_l2reg = l2reg
            best_estimator = ridge_regression_estimator

        print('l2reg: {:.2f} validation score: {:.4f} train score: {:.4f}'.format(l2reg, score, score_train))

    legend.append('Best ridge estimation')
    plt.plot(x_plot, best_estimator.predict(featurize(x_plot)), '-r')

    legend.append('Bayes estimation')
    plt.plot(x_plot, target_fn(x_plot), '-c')

    legend.append('data')
    plt.scatter(x_train, y_train, marker='^', c='g')

    plt.legend(legend)
    plt.title('Ridge regression')
    plt.grid()

    # print('=>' + str(len(legend)))

    # Visualize cost vs l2reg
    plt.subplot(2, 1, 2)
    plt.title('Ridge regression cost')
    plt.grid()
    plt.plot(l2reg_range, l2reg_costs, '-rx')
    f1.show()

    # Visualize weights
    f2 = plt.figure(2)
    plt.subplots_adjust(hspace=0.5)

    plt.subplot(3, 1, 1)
    plt.grid()
    plt.title('Weights without regularization')
    plt.bar(range(X_train[0].shape[0]), base_weights)

    plt.subplot(3, 1, 2)
    plt.grid()
    plt.title('Best regularized weights')
    plt.bar(range(X_train[0].shape[0]), ridge_regression_estimator.w_)

    plt.subplot(3, 1, 3)
    plt.grid()
    plt.title('Bayes regularized weights')
    plt.bar(range(X_train[0].shape[0]), coefs_true)
    f2.show()

    print('Best performance with l2reg: {:.4f} score: {:.4f}'.format(best_l2reg, best_score))
    print('length of vector w: {:} number of non-zero elements: {:}'
          .format(len(best_estimator.w_), np.count_nonzero(best_estimator.w_)))

    best_w_adjusted = np.copy(best_estimator.w_)
    print('length of vector w: {:} number of non-zero elements with tolerance 1e-6: {:}'
          .format(len(best_estimator.w_), np.count_nonzero(best_w_adjusted[best_w_adjusted > 1e-6])))

    plt.show()


if __name__ == '__main__':
  main()

