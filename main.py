"""
Name:       QNMs_for_Regression
Purpose:    Course project to compare Quasi-Newton Methods (SR1, BHHH, BFGS,
                L-BFGS)
Author:     Artem "Manneq" Arkhipov
Created:    14/10/2019
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import optimize
import cv2
import math
import os

# Set data for scipy.optimize.minimize callbacks
a_data, b_data, f_data = [], [], []
f_callback, x_callback, y_callback = None, None, None


def f_linear(x, args):
    """
        Linear regression function.
        param:  x: float scalar value
                args: tuple of float
        return: function result - scalar
    """
    return args[0] * x + args[1]


def f_rational(x, args):
    """
        Rational regression function.
        param:  x: float scalar value
                args: tuple of float
        return: function result - scalar
    """
    return args[0] / (1 + args[1] * x)


def error_mse(params, f, x, y):
    """
        MSE error for regression function.
        param:  params: tuple of float
                f: regression function - reference type
                x: numpy array of float (n, )
                y: numpy array of float (n, )
        return: function result - scalar
    """
    res = 0

    for k in range(x.size):
        res += (f(x[k], params) - y[k]) ** 2

    return res


def error_mse_fprime(params, f, x, y):
    """
        Jacobian for MSE error with scipy.optimize.approx_fprime.
        param:  params: tuple of float
                f: regression function - reference type
                x: numpy array of float (n, )
                y: numpy array of float (n, )
        return: numpy vector of float (2, )
    """
    return optimize.approx_fprime(params, error_mse, 1e-8, f, x, y)


def create_noisy_data(n, log_file):
    """
        Method to create noisy data.
        param:  n: number of sample - int
                log_file: log file path
        return: x: numpy vector of float (n, )
                y: numpy vector of float (n, )
    """
    alpha, betta = np.random.uniform(0, 1), np.random.uniform(0, 1)
    delta = np.random.normal(size=n)

    x, y = [], []

    for k in range(n):
        x.append(k / (n - 1))
        y.append(alpha * x[k] + betta + delta[k])

    log_file.write('Random noisy data created with: \n alpha: {} \n '
                   'betta: {} \n\n'.
                   format(alpha, betta))

    return np.array(x), np.array(y)


def compute_regression_function(f, x, args):
    """
        Method to compute regression function.
        param:  f: regression function - reference type
                x: perturbated x values - np.array of float (n, )
                args: tuple of float
        return: list of function values of float - (n, )
    """
    f_regression = []

    for i in range(x.size):
        f_regression.append(f(x[i], args))

    return f_regression


def draw_result(f_newtons, f_sr1, f_bhhh, f_bfgs, f_lbfgs, x, y, name,
                width=1600, height=900, dpi=96):
    """
        Method to plot results.
        param:  f_newtons: Newton function points - np.array of float (n, )
                f_sr1: SR1 function points - np.array of float (n, )
                f_bhhh: BHHH function points - np.array of float (n, )
                f_bfgs: BFGS function points - np.array of float (n, )
                f_lbfgs: L-BFGS function points - np.array of float (n, )
                x: perturbated x values - np.array of float (n, )
                y: perturbated y values - np.array of float (n, )
                name: name of regression - string
                width: plot width - 1600 inches
                height: plot height - 900 inches
                dpi: dpi of screen - 96 pix/inch
    """
    # Create normal plots
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.xlabel("x")
    plt.ylabel("f, y")
    plt.title(name)
    plt.ylim(min(y), max(y))
    plt.grid(True)
    plt.plot(x, y, '-b', label='Noisy samples')
    plt.plot(x, f_newtons, '-y', label='Newtons method', linewidth=3)
    plt.plot(x, f_sr1, '-m', label='SR1', linewidth=3)
    plt.plot(x, f_bhhh, '-c', label='BHHH', linewidth=3)
    plt.plot(x, f_bfgs, '-r', label='BFGS', linewidth=3)
    plt.plot(x, f_lbfgs, '-g', label='L-BFGS', linewidth=3)
    plt.legend(loc='upper center', ncol=3, frameon=True)
    plt.savefig("plots/" + name + ".png", dpi=dpi)
    plt.close()

    # Create zoomed plots
    name += " (zoomed)"

    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.xlabel("x")
    plt.ylabel("f, y")
    plt.title(name)
    plt.ylim(min([min(f_newtons), min(f_bfgs), min(f_lbfgs)]),
             max([max(f_newtons), max(f_bfgs), max(f_lbfgs)]))
    plt.grid(True)
    plt.plot(x, y, '-b', label='Noisy samples')
    plt.plot(x, f_newtons, '-y', label='Newtons method', linewidth=3)
    plt.plot(x, f_sr1, '-m', label='SR1', linewidth=3)
    plt.plot(x, f_bhhh, '-c', label='BHHH', linewidth=3)
    plt.plot(x, f_bfgs, '-r', label='BFGS', linewidth=3)
    plt.plot(x, f_lbfgs, '-g', label='L-BFGS', linewidth=3)
    plt.legend(loc='upper center', ncol=3, frameon=True)
    plt.savefig("plots/" + name + ".png", dpi=dpi)
    plt.close()

    return


def make_approximation_video(newtons_data, sr1_data, bhhh_data, bfgs_data,
                             lbfgs_data, args, name):
    """
        Method to make video of approximation.
        param:  newtons_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                sr1_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                bhhh_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                bfgs_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                lbfgs_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                args: list of 3 arguments:
                    1. f: regression function - reference type
                    2. x: perturbated x values - np.array of float (n, )
                    3. y: perturbated y values - np.array of float (n, )
                name: name of video - string
    """
    # Initialize paths to frames
    frames_paths = []

    # Cycle to draw frames
    for i in range(max(len(newtons_data[0]), len(sr1_data[0]),
                       len(bhhh_data[0]), len(bfgs_data[0]),
                       len(lbfgs_data[0]))):
        # Set frame name
        frame_name = "temp/" + name + " (iteration " + str(i + 1) + ").png"
        frames_paths.append(frame_name)

        # Make borders for np.array slices
        border_newtons = i + 1 if i + 1 < len(newtons_data[0]) else \
            len(newtons_data[0])
        border_sr1 = i + 1 if i + 1 < len(sr1_data[0]) else len(sr1_data[0])
        border_bhhh = i + 1 if i + 1 < len(bhhh_data[0]) else len(bhhh_data[0])
        border_bfgs = i + 1 if i + 1 < len(bfgs_data[0]) else len(bfgs_data[0])
        border_lbfgs = i + 1 if i + 1 < len(lbfgs_data[0]) else \
            len(lbfgs_data[0])

        # Draw approximation frame
        draw_approximation([newtons_data[0][0:border_newtons],
                            newtons_data[1][0:border_newtons],
                            newtons_data[2][0:border_newtons]],
                           [bhhh_data[0][0:border_bhhh],
                            bhhh_data[1][0:border_bhhh],
                            bhhh_data[2][0:border_bhhh]],
                           [sr1_data[0][0:border_sr1],
                            sr1_data[1][0:border_sr1],
                            sr1_data[2][0:border_sr1]],
                           [bfgs_data[0][0:border_bfgs],
                            bfgs_data[1][0:border_bfgs],
                            bfgs_data[2][0:border_bfgs]],
                           [lbfgs_data[0][0:border_lbfgs],
                            lbfgs_data[1][0:border_lbfgs],
                            lbfgs_data[2][0:border_lbfgs]],
                           args,
                           name + " (iteration " + str(i + 1) + ")")

    # Set fps
    fps = 5

    # Initialize frames
    frames = []
    for i in range(len(frames_paths)):
        img = cv2.imread(frames_paths[i])
        height, width, layers = img.shape
        size = (width, height)
        frames.append(img)

    # Set video writer
    out = cv2.VideoWriter("vids/" + name + ".avi",
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    # Write frames to video
    for i in range(len(frames)):
        out.write(frames[i])

    # Release writer
    out.release()

    # Delete frames
    for i in range(len(frames_paths)):
        os.remove(frames_paths[i])

    return


def draw_approximation(newtons_data, sr1_data, bhhh_data, bfgs_data,
                       lbfgs_data, args, name,
                       width=1600, height=900, dpi=96):
    """
        Method to make approximation plot.
        param:  newtons_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                sr1_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                bhhh_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                bfgs_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                lbfgs_data: list of 3 elements:
                    1. list of first params
                    2. list of second params
                    3. error func values
                name: name of plot - string
                width: plot width - 1600 inches
                height: plot height - 900 inches
                dpi: dpi of screen - 96 pix/inch
    """
    # Set data for plane
    a = np.linspace(-1.1, 3, 20)
    b = np.linspace(-1.1, 3, 20)

    # Create mesh grid
    a_mesh, b_mesh = np.meshgrid(a, b)
    # Create plane data
    error_mse_points = error_mse([a_mesh, b_mesh], args[0], args[1], args[2])

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(-1.1, 3)
    ax.set_ylim3d(-1.1, 3)
    ax.set_zlim3d(np.amin(error_mse_points), np.amax(error_mse_points))
    ax.plot_wireframe(a_mesh, b_mesh, error_mse_points, color='black')
    ax.plot(newtons_data[0], newtons_data[1], newtons_data[2], '-y',
            linewidth=3, label='Newtons method')
    ax.plot(sr1_data[0], sr1_data[1], sr1_data[2], '-m',
            linewidth=3, label='SR1')
    ax.plot(bhhh_data[0], bhhh_data[1], bhhh_data[2], '-c',
            linewidth=3, label='BHHH')
    ax.plot(bfgs_data[0], bfgs_data[1], bfgs_data[2], '-r', linewidth=3,
            label='BFGS')
    ax.plot(lbfgs_data[0], lbfgs_data[1], lbfgs_data[2], '-g', linewidth=3,
            label='L-BFGS')
    ax.legend(loc='center right', frameon=True)
    ax.set_title(name)
    plt.savefig("temp/" + name + ".png", dpi=dpi)
    plt.close()

    return


def callback_function(params):
    """
        Callback function for scipy.optimize.minimize.
        param:  params: tuple of float
    """
    global f_callback, x_callback, y_callback

    a_data.append(params[0])
    b_data.append(params[1])
    f_data.append(error_mse(params, f_callback, x_callback,
                            y_callback))


def clear_callbacks():
    """
        Method to clear callbacks.
    """
    a_data.clear()
    b_data.clear()
    f_data.clear()

    return


def write_callbacks(log_file):
    """
        Method to write callbacks to log file.
        param:  log_file: log file path
    """
    log_file.write('{0:4s}   {1:9s}   {2:9s}   {3:9s}\n'.
                   format('Iteration', ' a', ' b',  'D(a, b)'))

    for i in range(len(a_data)):
        log_file.write('{0:9d}   {1:9f}   {2:9f}   {3:9f}\n'.
                       format(i + 1, a_data[i], b_data[i], f_data[i]))

    log_file.write('\n')

    return


def sr1_method(initial_approximation, args, eps=1e-3):
    """
        Newton's method with SR1 strategy by scipy.optimize.SR1.
        param:  initial_approximation: initial approximation - tuple of float
                args: list length of 3 arguments:
                    1.  f: regression function - reference type
                    2.  x: np.array of float - (n, )
                    3.  y: np.array of float - (n, )
                eps: estimated error - 1e-3
        return: xn_1: tuple of float params
    """
    # Initialize SR1 hessian
    hessian = optimize.SR1()
    hessian.initialize(n=2, approx_type='inv_hess')

    xn = initial_approximation

    def hessian_dot_fprime(xn, f, x, y):
        """
            Method to return multiplication of SR1 hessian and jacobian
                of MSE function.
            param:  xn: tuple of float parameters
                    f: regression function - reference type
                    x: np.array of float - (n, )
                    y: np.array of float - (n, )
            return: multiplication of matrix SR1 hessian and jacobian
                of MSE function
        """
        return hessian.dot(error_mse_fprime(xn, f, x, y))

    while True:
        d_xn = hessian_dot_fprime(xn, args[0], args[1], args[2])

        # Compute a step size using scipy.optimize.line_search to satisfy
        # the Wolf conditions
        step = optimize.line_search(error_mse, hessian_dot_fprime,
                                    np.r_[xn[0], xn[1]],
                                    -np.r_[d_xn[0], d_xn[1]],
                                    args=(args[0], args[1], args[2]))
        step = step[0]

        if step is None:
            step = 0

        # Compute new params
        xn_1 = xn - step * d_xn

        # Write a callback
        callback_function(xn_1)

        if abs(error_mse(xn, args[0], args[1], args[2]) -
               error_mse(xn_1, args[0], args[1], args[2])) < eps:
            break

        # Update SR1 hessian
        hessian.update(xn_1 - xn,
                       error_mse_fprime(xn_1, args[0], args[1], args[2]) -
                       error_mse_fprime(xn, args[0], args[1], args[2]))
        xn = xn_1

    return xn_1


def bhhh_algorithm(initial_approximation, args, eps=1e-3):
    """
        BHHH algorithm.
        param:  initial_approximation: initial approximation - tuple of float
                args: list length of 3 arguments:
                    1.  f: regression function - reference type
                    2.  x: np.array of float - (n, )
                    3.  y: np.array of float - (n, )
                eps: estimated error - 1e-3
        return: xn_1: tuple of params - float
    """
    xn = initial_approximation

    def a_dot_error_fprime(xn, f, x, y):
        """
            Method to return multiplication of A and jacobian of MSE function.
            param:  xn: tuple of float parameters
                    f: regression function - reference type
                    x: np.array of float - (n, )
                    y: np.array of float - (n, )
            return: Multiplication of matrix A and jacobian of MSE function
        """
        return a.dot(error_mse_fprime(xn, f, x, y))

    def ln_regression(params, f, x, y):
        """
            Method to return natural logarithm of square difference of
                regression function and noisy samples y.
            param:  params: tuple of float parameters
                    f: regression function - reference type
                    x: np.array of float - (n, )
                    y: np.array of float - (n, )
            return: natural logarithm of square difference of regression
                function and noisy samples y
        """
        return math.log((f(x, params) - y) ** 2)

    def ln_regression_fprime(xn, f, x, y):
        """
            Method to return jacobian of natural logarithm of square
                difference of regression function and noisy samples y.
            param:  xn: tuple of float parameters
                    f: regression function - reference type
                    x: np.array of float - (n, )
                    y: np.array of float - (n, )
            return: jacobian of natural logarithm of square difference of
                regression function and noisy samples y
        """
        return optimize.approx_fprime(xn, ln_regression, 1e-8, f, x, y)

    def compute_a(xn, f, x, y):
        """
            Method to compute inverted matrix A.
            param:  xn: tuple of float parameters
                    f: regression function - reference type
                    x: np.array of float - (n, )
                    y: np.array of float - (n, )
            return: inverted matrix A
        """
        res = 0

        for k in range(args[1].size):
            res += ln_regression_fprime(xn, f, x[k], y[k]).reshape(1, 2) * \
                   ln_regression_fprime(xn, f, x[k], y[k]).reshape(2, 1)

        return np.linalg.inv(res)

    while True:
        a = compute_a(xn, args[0], args[1], args[2])

        d_xn = a_dot_error_fprime(xn, args[0], args[1], args[2])

        # Compute a step size using scipy.optimize.line_search to satisfy
        # the Wolf conditions
        step = optimize.line_search(error_mse, a_dot_error_fprime,
                                    np.r_[xn[0], xn[1]],
                                    -np.r_[d_xn[0], d_xn[1]],
                                    args=(args[0], args[1], args[2]))
        step = step[0]

        if step is None:
            step = 0

        # Compute new params
        xn_1 = xn - step * d_xn

        # Write a callback
        callback_function(xn_1)

        # Check to estimated error
        if abs(error_mse(xn, args[0], args[1], args[2]) -
               error_mse(xn_1, args[0], args[1], args[2])) < eps:
            break

        xn = xn_1

    return xn_1


def do_regressions(f, x, y, initial_approximation, log_file, name):
    """
        Method to compute all approximations for regression function:
            1. Newton's method
            2. Newton's method with SR1 strategy
            3. BHHH algorithm
            4. BFGS algorithm
            5. L-BFGS algorithm
        param:  f: regression function - reference type
                x: perturbated x samples - np.array of float (n, )
                y: perturbated y samples - np.array of float (n, )
                initial_approximation: initial approximation - tuple of float
                log_file: log file path
                name: name of regression
    """
    # Define data for callbacks
    global f_callback
    f_callback = f

    log_file.write(name + '\n')

    # Newton's method
    log_file.write('Newtons method: \n')
    result = optimize.minimize(error_mse, initial_approximation,
                               method='Newton-CG', jac=error_mse_fprime,
                               args=(f, x, y),
                               callback=callback_function,
                               options={'eps': 1e-3})
    f_newton = compute_regression_function(f, x, result.x)
    newton_data = \
        [np.array(a_data), np.array(b_data), np.array(f_data)]

    # Manage callbacks data
    write_callbacks(log_file)
    clear_callbacks()

    # Newton's method with SR1 strategy
    log_file.write('SR1 method: \n')
    result = sr1_method(initial_approximation, (f, x, y))
    f_sr1 = compute_regression_function(f, x, result)
    sr1_data = \
        [np.array(a_data), np.array(b_data), np.array(f_data)]

    # Manage callbacks data
    write_callbacks(log_file)
    clear_callbacks()

    # BHHH algorithm
    log_file.write('Berndt-Hall-Hall-Hausman algorithm: \n')
    result = bhhh_algorithm(initial_approximation, (f, x, y))
    f_bhhh = compute_regression_function(f, x, result)
    bhhh_data = \
        [np.array(a_data), np.array(b_data), np.array(f_data)]

    # Manage callbacks data
    write_callbacks(log_file)
    clear_callbacks()

    # BFGS algorithm
    log_file.write('Broyden-Fletcher-Goldfarb-Shanno algorithm: \n')
    result = optimize.minimize(error_mse, initial_approximation, method='BFGS',
                               args=(f, x, y),
                               callback=callback_function,
                               options={'eps': 1e-3})
    f_bfgs = compute_regression_function(f, x, result.x)
    bfgs_data = [np.array(a_data), np.array(b_data), np.array(f_data)]

    # Manage callbacks data
    write_callbacks(log_file)
    clear_callbacks()

    # L-BFGS algorithm
    log_file.write('Limited-memory BFGS:\n')
    result = optimize.minimize(error_mse, initial_approximation,
                               method='L-BFGS-B', callback=callback_function,
                               args=(f, x, y),
                               options={'eps': 1e-3})
    f_lbfgs = compute_regression_function(f, x, result.x)
    lbfgs_data = [np.array(a_data), np.array(b_data), np.array(f_data)]

    # Manage callbacks data
    write_callbacks(log_file)
    clear_callbacks()

    # Draw regression results
    draw_result(f_newton, f_sr1, f_bhhh, f_bfgs, f_lbfgs, x, y, name)
    # Create approximation process video
    make_approximation_video(newton_data, sr1_data, bhhh_data, bfgs_data,
                             lbfgs_data, (f, x, y), 'MSE with ' + name)

    log_file.write('\n')

    return


def main():
    """
        Main function.
    """
    # Open log file
    log_file = open('log.txt', 'w')

    x, y = create_noisy_data(101, log_file)

    # Define data for callbacks
    global x_callback, y_callback
    x_callback, y_callback = x, y

    # Do Linear regressions
    do_regressions(f_linear, x, y, ([3., 3.]), log_file,
                   'Linear regression')

    # Do Rational regressions
    do_regressions(f_rational, x, y, ([3., 3.]), log_file,
                   'Rational regression')

    log_file.close()


if __name__ == "__main__":
    main()
