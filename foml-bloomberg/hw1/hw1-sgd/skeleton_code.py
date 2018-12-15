import pandas as pd
import logging
import numpy as np
import sys
import matplotlib as mpl
if (sys.platform.startswith('linux')):
    mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

### Assignment Owner: Alex Nikiforov

#######################################
#### Normalization


def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.

    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """
    # TODO
    (train_num_instances, train_num_features) = train.shape
    train_num_features_min = np.zeros((train_num_features), dtype=float)
    train_num_features_max = np.zeros((train_num_features), dtype=float)

    for col in np.arange(train_num_features):
        cur_col = train[:, col]
        var = np.var(cur_col)
        if np.allclose(var, 0.0) == False:
            train_num_features_min[col] = np.amin(cur_col)
            train_num_features_max[col] = np.amax(cur_col) - train_num_features_min[col]
        else: 
            train_num_features_min[col] = 0.0
            train_num_features_max[col] = 1.0
            
    # print(str(train))
    # print('min: ' + str(train_num_features_min))
    # print('diff: ' + str(train_num_features_max))

    train_normalized = (train - train_num_features_min) / train_num_features_max
    test_normalized = (test - train_num_features_min) / train_num_features_max

    # (test_num_instances, test_num_features) = test.shape
    return train_normalized, test_normalized


########################################
#### The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    # loss = 0 #initialize the square_loss

    # print('X: ' + str(X.shape) + ' y: ' + str(y.shape) + ' theta: ' + str(theta.shape))
    tmp = X.dot(theta) - y.reshape(X.shape[0], 1)
    loss = tmp.T.dot(tmp) / X.shape[0]

    return loss

########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """

    return 2 * X.T.dot(X.dot(theta) - y) / float(X.shape[0])


###########################################
### Gradient Checker
#Getting the gradient calculation correct is often the trickiest part
#of any gradient-based optimization algorithm.  Fortunately, it's very
#easy to check that the gradient calculation is correct using the
#definition of gradient.
#See http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4):
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions:
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1)

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by:
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error

    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    #TODO

#################################################
### Generic Gradient Checker
def generic_gradient_checker(X, y, theta, objective_func, gradient_func, epsilon=0.01, tolerance=1e-4):
    """
    The functions takes objective_func and gradient_func as parameters. And check whether gradient_func(X, y, theta) returned
    the true gradient for objective_func(X, y, theta).
    Eg: In LSR, the objective_func = compute_square_loss, and gradient_func = compute_square_loss_gradient
    """
    #TODO


####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.01, num_iter=1000, check_gradient=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features), dtype=float)  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1, dtype=float) #initialize loss_hist
    grad_hist = np.zeros(num_iter+1, dtype=float)
    theta = np.zeros(num_features, dtype=float) #initialize theta

    for i in xrange(num_iter + 1):
        print(str(theta))
        loss_hist[i] = compute_square_loss(X, y, theta)
        theta_hist[i] = theta
        # print('' + str(theta))
        # print('' + str(compute_square_loss_gradient(X, y, theta)))
        grad = compute_square_loss_gradient(X, y, theta)
        grad_hist[i] = np.linalg.norm(grad)
        theta = theta - alpha * grad

    # print(str(loss_hist))
    # plt.grid()
    # plt.plot(loss_hist)
    # plt.savefig('testfigure.png', dpi=100)
    # plt.show(block=True)

    return loss_hist

####################################
###Q2.4b: Implement backtracking line search in batch_gradient_descent
###Check http://en.wikipedia.org/wiki/Backtracking_line_search for details
# def backtracking_line_search(X, y, loss, gradient)


def compute_regularized_square_loss(X, y, theta, lambda_reg):
    tmp = X.dot(theta) - y
    loss = tmp.T.dot(tmp) / X.shape[0] + lambda_reg * theta.T.dot(theta)
    return loss

###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    return (2 * X.T.dot(X.dot(theta) - y) / float(X.shape[0])) + 2 * lambda_reg * theta

###################################################
### Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.01, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    grad_hist = np.zeros(num_iter+1, dtype=float)
    loss_hist = np.zeros(num_iter+1, dtype=float) #Initialize loss_hist

    for i in xrange(num_iter + 1):
        print(str(theta))
        loss_hist[i] = compute_regularized_square_loss(X, y, theta=theta, lambda_reg=lambda_reg)
        theta_hist[i] = theta
        # print('' + str(theta))
        # print('' + str(compute_square_loss_gradient(X, y, theta)))
        grad = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
        grad_hist[i] = np.linalg.norm(grad)
        theta = theta - alpha * grad

    # print(str(loss_hist))
    # plt.grid()
    # plt.plot(grad_hist)
    # plt.savefig('regularized_grad_descent.png', dpi=100)

    return loss_hist

#############################################
## Visualization of Regularized Batch Gradient Descent
##X-axis: log(lambda_reg)
##Y-axis: square_loss

#############################################
### Stochastic Gradient Descent
def stochastic_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    In this question you will implement stochastic gradient descent with a regularization term

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set

    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features)
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones((num_features, 1)) #Initialize theta

    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter * X.shape[0], dtype=float) #Initialize loss_hist
    grad_hist = np.zeros(num_iter, dtype=float)

    # print('theta:' + str(theta))
    # print(str(X))
    # print(str(y))
    # print(str(batch))
    for epoch in range(num_iter):
        idx = range(X.shape[0])
        np.random.shuffle(idx)
        # alpha = 1.0 / (epoch + 1)
        # print('idx:' + str(idx)) 
        for i in range(len(idx)):
            loss_hist[epoch * num_iter + i] = compute_square_loss(X, y, theta)

            X_item = X[i, :]
            y_item = y[i]
            # print(str(i))
            # print(str(X_item))
            # print(str(y_item))

            grad = X_item * (X_item.dot(theta) - y_item) #+ 0 * lambda_reg * theta.dot(theta.T)
            # print('grad:' + str(grad))

            theta = theta - alpha * grad.reshape(X.shape[1], 1)
            # print('theta:' + str(theta))

    return loss_hist

def plot_data(X, y):
    cov_mat = np.cov(X.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    matrix_w = np.hstack((eig_pairs[0][1].reshape(X.shape[1], 1),
                      eig_pairs[1][1].reshape(X.shape[1], 1)))
    Ak = X.dot(matrix_w)
   
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(np.real(Ak[:, 0]), np.real(Ak[:, 1]), y)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.savefig('testfigure.png')

################################################
### Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value) and/or objective_function_value

def regularized_grad_descent_study(X, y):
    lambdas = [np.power(10, float(-7)), np.power(10, float(-5)), np.power(10, float(-3)), np.power(10, float(-1)), 1, 10]
    lambda_size = len(lambdas)

    plt.grid()

    for i in range(lambda_size):
        loss_hist = regularized_grad_descent(X, y, alpha=.01, lambda_reg=lambdas[i])
        plt.plot(np.log(loss_hist), label=str(lambdas[i]))

    plt.legend(loc='right', ncol=2, mode="expand", shadow=True, fancybox=True)
    plt.savefig('regurized_study.png', dpi=100)

def main():
    #Loading the dataset
    print('loading the dataset')

    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    if True:
        print('Split into Train and Test. Shape: ' + str(X.shape))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)
    else:
        # small data
        X_train = np.array((([1, 2, 3], [4, 5, 6])), dtype = float)
        X_test = np.array(([1, 2, 3], [4, 5, 6]), dtype = float)
        y_train = np.array([1, 1], dtype = float)
        theta = np.array([1, -1, -1, 1], dtype = float)

    print("Scaling all to [0, 1]")
    print('X shape: ' + str(X_train.shape) + ' y shape: ' + str(y_train.shape))

    X_train, X_test = feature_normalization(X_train, X_test)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) # Add bias term

    # hw1 2.2.5
    # res_sl = compute_square_loss(X_train, y_train, theta)
    # res_sl = compute_square_loss(X_train, np.dot(X_train, theta), theta)
    # print('compute_square_loss: ' + str(res_sl))

    # hw1 2.2.6
    # res_grad = compute_square_loss_gradient(X_train, y_train, theta)
    # res = compute_square_loss_gradient(X, np.dot(X, theta), theta)
    # print('compute_square_loss_gradient: ' + str(res_grad))
    
    # TODO: type on the HW page 3
    # hw1 2.4.1
    # alpha = 0.04
    # grad_desc = batch_grad_descent(X_train, y_train, alpha=alpha)

    # hw1 2.5.2
    # res_rsl = compute_regularized_square_loss_gradient(X_train, y_train, theta, 0)
    # print('compute_square_loss: ' + str(res_rsl))

    # hw1 2.5.3
    # ridge_desc = regularized_grad_descent(X_train, y_train, alpha=alpha, lambda_reg=2)

    # hw1 2.5.7
    # regularized_grad_descent_study(X_train, y_train)

    # some comparison
    # plt.grid()
    # plt.plot(np.log(grad_desc), label='Regression')
    # plt.plot(np.log(ridge_desc), label='Ridge regression')
    # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
    #        ncol=2, mode="expand", borderaxespad=0.)
    # plt.savefig('result.png', dpi=100)

    # just plot
    # plot_data(X_train, y_train)

    # hw 2.6
    stochastic_desc = stochastic_grad_descent(X_train, y_train, alpha=0.05, lambda_reg=0, num_iter=10)
    plt.grid()
    plt.plot(stochastic_desc[0:200], label='Stochastic descent')
    plt.savefig('stochastic.png', dpi=100)

if __name__ == "__main__":
    main()
