"""
Demonstrate gradient descent for multiple linear regression
"""
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("./deeplearning.mplstyle")
np.set_printoptions(precision=2)
x_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])


def predict_single_loop(x, w, b):
    "Predict a single sample"
    return np.dot(x, w) + b


def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_cost = 0.0
    for i in range(m):
        f_wb_i = predict_single_loop(x[i], w, b)
        total_cost += (y[i] - f_wb_i)**2
    return total_cost/(2*m)


def compute_gradient(x, y, w, b):
    m, n = x.shape
    dj_dw = np.zeros(n,)
    dj_db = 0.0
    for i in range(m):
        error = np.dot(x[i], w) + b - y[i]
        for j in range(n):
            dj_dw[j] += error * x[i, j]
        dj_db += error
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db


tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, learning_rate, number_iterations):
    w = copy.deepcopy(w_in)
    b = b_in
    m,n = x.shape
    cost_history = []
    for i in range (number_iterations):
        temp_dj_dw, temp_dj_db = gradient_function(x, y, w, b)
        w = w - learning_rate * temp_dj_dw
        b = b - learning_rate * temp_dj_db
        cost_history.append(cost_function(x, y, w, b))
        if i% math.ceil(number_iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {cost_history[-1]:8.2f}   ")
        
    return w, b, cost_history #return final w,b and J history for graphing

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = x_train.shape
for i in range(m):
    print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
