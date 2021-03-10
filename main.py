import numpy as np
import random


def gradient(matrix, vector, omega):
    return np.add((np.dot((matrix.T + matrix), omega)), vector)


def get_j(a, b, c, x):
    return c + np.add(np.dot(b.T, x), np.dot(np.dot(x.T, a), x))


def gradient_descent(tolerance, learning_rate, max_iter, a, b, c, x0):
    j = get_j(a, b, c, x0)
    count = 0
    while count < max_iter and j > tolerance:
        x = x0 - np.dot(gradient(a, b, x0), learning_rate)
        count += 1
        x0 = x
        j = get_j(a, b, c, x0)
    return x0


def hessian(a):
    return np.add(a, a.T)


def newton(tolerance, max_iter, a, b, c, x0):
    j = get_j(a, b, c, x0)
    inv_hess = np.linalg.inv(hessian(a))
    count = 0
    while count < max_iter and j > tolerance:
        x = x0 - np.dot(gradient(a, b, x0), inv_hess)
        count += 1
        x0 = x
        j = get_j(a, b, c, x0)
    return x0


# v = [1, 2, 3]
# m = np.array([[1, 2, 3], [3, 4, 5], [7, 8, 9]])
# w = [3, 3, 5]

# print(gradient(m, v, w))
# xx = newton(0.00001, 10, np.array([[1, 0], [0, 1]]), np.array([1, 0]), 1, np.array([3, 1]))
# print(xx)
# print(get_j(np.array([[1, 0], [0, 1]]), np.array([1, 0]), 1, xx))
# print(gradient_descent(0.00001, 0.5, 10, np.array([[1, 0], [0, 1]]), np.array([1, 0]), 1, np.array([3, 1])))

# print('Enter Numbers')
# entries = list(map(float, input().split()))
# x_now = np.array(entries)
# print(x_now)
#
# print('Enter Numbers')
# entries = list(map(float, input().split()))
# b = np.array(entries)
# print(b)
#
# print('Enter Numbers')
# entries = list(map(float, input().split()))
# a = np.array(entries).reshape(2, 2)
# print(a)
#
# print('Enter range')
# entries = list(map(float, input().split()))
# var = []
# for i in range(len(b)):
#     var.append(random.uniform(entries[0], entries[1]))
#
# print(var)


def get_func_coeff(dim):
    print('Enter Numbers for vector b separated by space')
    entries = list(map(int, input().split()))
    b = np.array(entries)
    size_b = b.size

    if dim != size_b:
        raise Exception('Size of vector b must match d')

    print('Enter Numbers for matrix A in order of each row separated by space')
    entries = list(map(int, input().split()))
    a = np.array(entries).reshape(size_b, size_b)

    print('Enter c')
    c = float(input())

    return a,b,c
def get_param():
    print('Enter tolerance')
    tol = float(input())
    print('Enter max iterations')
    max_iter = int(input())
    print('Enter learning rate (Enter 0 is not applicable)')
    learning_rate =float(input())

    return tol,max_iter,learning_rate
def run_methods(a,b,c,x):
    tol,max_iter,learning_rate = get_param()
    print('Enter 1 to run gradient Descent OR 2 to run Newtons method OR 3 to run both')
    path = int(input())
    if  path == 1 or path == 3:
        sol = gradient_descent(tol,learning_rate,max_iter,a,b,c,x)
        print('Solution found by gradient descent:')
        print(sol)
    if path == 2 or path == 3:
        sol = newton(tol, max_iter, a, b, c, x)
        print('Solution found by Newton:')
        print(sol)
    else:
        print('Incorrect Option chosen')



# MAIN INTERFACE

print('Enter 1 to manually add all data OR 2 to choose range for x/ Batch mode')
path = int(input())
if path == 1:
    print('Enter dimension')
    dim = int(input())
    print('Enter Numbers for x separated by space')
    entries = list(map(int, input().split()))
    x_now = np.array(entries)

    a,b,c = get_func_coeff(dim)

    run_methods(a,b,c,x_now)

elif path == 2:
    print('Enter dimension')
    dim = int(input())
    print('Enter 1 to choose range for x OR 2 for Batch mode') # Batch mode not implemented yet
    path = int(input())
    if path == 1:
        print('Enter range for x separated by space')
        entries = list(map(float, input().split()))
        x = []
        for i in range(dim):
            x.append(random.uniform(entries[0], entries[1]))
        a,b,c = get_func_coeff(dim)
        run_methods(a,b,c,np.array(x))
else:
    print('Incorrect option entered')

