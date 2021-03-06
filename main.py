import numpy as np
import random


def gradient(matrix, vector, omega):
    return np.add((np.dot((matrix.T + matrix), omega)),
                  vector)  # Gradient of general quadratic function (A trans + A)w +b


def get_j(a, b, c, x):  # Calculating value of function at x
    return c + np.add(np.dot(b.T, x), np.dot(np.dot(x.T, a), x))


def gradient_descent(tolerance, learning_rate, max_iter, a, b, c, x0):
    j = get_j(a, b, c, x0)
    count = 0
    while count < max_iter and j > tolerance:
        x = x0 - np.dot(gradient(a, b, x0), learning_rate)
        count += 1
        x0 = x
        j = get_j(a, b, c, x0)

    return x0,j


def hessian(a):
    return np.add(a, a.T) # Calculating hessian of quadratic function A + A^T


def newton(tolerance, max_iter, a, b, c, x0):
    j = get_j(a, b, c, x0)
    inv_hess = np.linalg.inv(hessian(a))   # Inverse Hessian as required in the Newton Method
    count = 0
    while count < max_iter and j > tolerance:
        x = x0 - np.dot(gradient(a, b, x0), inv_hess)
        count += 1
        x0 = x
        j = get_j(a, b, c, x0)
    return x0,j


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
        if np.any(np.linalg.eigvals(a) <= 0): # Checking if given matrix is positive-definite
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
    while tol < 0:
        print('Incorrect input. Enter a non-negative number')
        tol = float(input())
    print('Enter limit of iterations')
    max_iter = int(input())
    while max_iter <= 0:
        print('Incorrect input. Enter a positive number')
        max_iter = int(input())
    print('Enter learning rate (Enter 0 if not applicable)')
    learning_rate = float(input())
    while learning_rate < 0:
        print('Incorrect input. Enter a positive number')
        learning_rate = float(input())
    return tol, max_iter, learning_rate


def run_methods(a, b, c, x):
    solution = {}
    tol, max_iter, learning_rate = get_param()
    print('Enter 1 to run gradient Descent OR 2 to run Newtons method OR 3 to run both')
    while True:
        path = int(input())
        if path == 1 or path == 3:
            if learning_rate != 0:
                sol, j = gradient_descent(tol, learning_rate, max_iter, a, b, c, x)
                solution['Solution by Gradient Descent:  '] = sol
                solution['J(x) by Gradient Descent: '] = j
            else:
                print('Learning rate is 0, hence Gradient Descent cannot be run')
        if path == 2 or path == 3:
            sol,j = newton(tol, max_iter, a, b, c, x)
            solution['Solution by Newton Method: '] = sol
            solution['J(x) by Newton Method: '] = j
        if path != 1 and path != 2 and path != 3:
            print('Incorrect option chosen')
            continue
        return solution

def check_learning_rate(learning_rate):
    if learning_rate == 0:
        print('Learning rate was 0, please enter new non-zero learning rate to run Batch Mode')
        learning_rate = float(input())
        while learning_rate < 0:
            print('Learning rate was < 0, please enter new rate to run Batch Mode')
            learning_rate = float(input())
    return learning_rate

def batch_mode(coeff, params, dim, n, entries):
    a, b, c = coeff
    tol, max_iter, learning_rate = params
    learning_rate = check_learning_rate(learning_rate)
    sol_vec_newton = []
    j_vec_newton = []
    sol_vec_grad = []
    j_vec_grad = []
    for i in range(n):
        x = []
        for i in range(dim):
            x.append(random.uniform(entries[0], entries[1]))
        sol_vec_newton.append((newton(tol, max_iter, a, b, c, np.array(x)))[0])
        j_vec_newton.append((newton(tol, max_iter, a, b, c, np.array(x)))[1])
        sol_vec_grad.append((gradient_descent(tol, learning_rate, max_iter, a, b, c, np.array(x)))[0])
        j_vec_grad.append((gradient_descent(tol, learning_rate, max_iter, a, b, c, np.array(x)))[1])
    print('STD and MEAN for Newton ')
    print(np.std(sol_vec_newton, axis=0))
    print(np.mean(sol_vec_newton, axis=0))
    print('\nSTD and MEAN for J(x) by Newton \n')
    print(np.std(j_vec_newton, axis=0))
    print(np.mean(j_vec_newton, axis=0))
    print('\nSTD and MEAN for for Gradient Descent \n')
    print(np.std(sol_vec_grad, axis=0))
    print(np.mean(sol_vec_grad, axis=0))
    print('\nSTD and MEAN for J(x) by Gradient Descent \n')
    print(np.std(j_vec_grad, axis=0))
    print(np.mean(j_vec_grad, axis=0))



# MAIN INTERFACE

print('Enter 1 to manually add all data OR 2 to choose range for x / Batch mode')
while True:
    path = int(input())
    if path == 1:
        print('Enter dimension')
        dim = int(input())
        print('Enter numbers for x separated by space')
        entries = list(map(int, input().split()))
        x_now = np.array(entries)

        a, b, c = get_func_coeff(dim)

        solution = run_methods(a, b, c, x_now)
        for i in solution:
            print(i, solution[i])
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
                x = []
                for i in range(dim):
                    x.append(random.uniform(entries[0], entries[1]))
                a, b, c = get_func_coeff(dim)
                solution = run_methods(a, b, c, np.array(x))
                for i in solution:
                    print(i, solution[i])
                break
            elif path == 2:
                print('Enter n - number of times to run methods')
                n = int(input())
                batch_mode(get_func_coeff(dim), get_param(), dim, n, entries)
                break
            print('Incorrect option entered try again')
        break
    print('Incorrect option entered please try again')

