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
    print('Enter numbers for vector b separated by space')
    entries = list(map(int, input().split()))
    b = np.array(entries)
    while b.size != dim:
        print(f'Size of vector b must be equal to {dim}. Enter numbers for vector b separated by space')
        entries = list(map(int, input().split()))
        b = np.array(entries)

    print('Enter numbers for matrix A in order of each row separated by space')
    entries = list(map(int, input().split()))
    while True:
        if len(entries) != dim * dim:
            print(f'Matrix size must be equal to {dim}x{dim}. Enter {dim * dim} numbers')
            entries = list(map(int, input().split()))
            continue
        a = np.array(entries).reshape(b.size, b.size)
        if np.any(np.linalg.eigvals(a) <= 0):
            print(f'Matrix must be positive-definite. Enter {dim * dim} numbers')
            entries = list(map(int, input().split()))
            continue
        else:
            break

    print('Enter c')
    c = float(input())
    return a, b, c


def get_param():
    print('Enter tolerance')
    tol = float(input())
    print('Enter max iterations')
    max_iter = int(input())
    print('Enter learning rate (Enter 0 is not applicable)')
    learning_rate = float(input())

    return tol, max_iter, learning_rate


def run_methods(a, b, c, x):
    tol, max_iter, learning_rate = get_param()
    print('Enter 1 to run gradient Descent OR 2 to run Newtons method OR 3 to run both')
    path = int(input())
    if path == 1 or path == 3:
        sol = gradient_descent(tol, learning_rate, max_iter, a, b, c, x)
        print('Solution found by gradient descent:')
        print(sol)
    if path == 2 or path == 3:
        sol = newton(tol, max_iter, a, b, c, x)
        print('Solution found by Newton:')
        print(sol)
    if path != 1 and path != 2 and path != 3:
        print('Incorrect Option chosen')

def batch_mode(coeff, params, dim, n, entries):
    a, b, c = coeff
    tol, max_iter, learning_rate = params
    sol_vec_newton = []
    sol_vec_grad = []
    for i in range(n):
        x = []
        for i in range(dim):
            x.append(random.uniform(entries[0], entries[1]))
        sol_vec_newton.append(newton(tol, max_iter, a, b, c, np.array(x)))
        sol_vec_grad.append(gradient_descent(tol, learning_rate, max_iter, a, b, c, np.array(x)))
        # run_methods(a, b, c, np.array(x))
    print('STD and MEAN for Newton \n')
    print(np.std(sol_vec_newton, axis=0))
    print(np.mean(sol_vec_newton, axis=0))
    print('STD and MEAN for Gradient Descent \n')
    print(np.std(sol_vec_grad, axis=0))
    print(np.mean(sol_vec_grad, axis=0))
    # print(np.std(sol_vec,axis=1))

# MAIN INTERFACE

print('Enter 1 to manually add all data OR 2 to choose range for x/ Batch mode')
while True:
    path = int(input())
    if path == 1:
        print('Enter dimension')
        dim = int(input())
        print('Enter numbers for x separated by space')
        entries = list(map(int, input().split()))
        x_now = np.array(entries)

        a, b, c = get_func_coeff(dim)

        run_methods(a, b, c, x_now)

        break

    elif path == 2:
        print('Enter dimension')
        dim = int(input())
        print('Enter range for x separated by space')
        entries = list(map(float, input().split()))
        print('Enter 1 to run methods OR 2 for Batch mode')  
        while True:
            path = int(input())
            if path == 1:
                # print('Enter range for x separated by space')
                # entries = list(map(float, input().split()))
                x = []
                for i in range(dim):
                    x.append(random.uniform(entries[0], entries[1]))
                a, b, c = get_func_coeff(dim)
                run_methods(a, b, c, np.array(x))
                break
            elif path == 2:
                print('Enter n - number of time to run methods')
                n = int(input())
                batch_mode(get_func_coeff(dim), get_param(), dim, n, entries)
                break
            print('Incorrect option entered try again')
        break
    print('Incorrect option entered please try again')








