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

    print('Original: ' + str(x_train[0]) + ' Featurized: ' + str(X_train[0]))
    print('Featurized shape: ' + str(X_train[0].shape))

    ridge_regression_estimator = MyRidge(l2reg=20)
    ridge_regression_estimator.fit(X_train, y_train)

    plt.scatter(x_train, y_train)
    x_plot = np.linspace(0, 1, 1000)
    y_plot = ridge_regression_estimator.predict(featurize(x_plot))
    plt.plot(x_plot, y_plot, 'red')

    plt.grid()
    plt.show()

if __name__ == '__main__':
  main()
