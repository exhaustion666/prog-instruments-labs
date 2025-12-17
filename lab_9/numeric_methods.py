import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from prettytable import PrettyTable
from scipy.optimize import fsolve


FIRST_MATRIX = np.random.randint(-8, 9, (3, 3))
SECOND_MATRIX = np.array([
    [8.2, 3.2, 14.2, 14.8],
    [5.6, 12, 15, 6.4],
    [5.7, 3.6, 12.4, 2.3],
    [6.8, 13.2, 6.3, 8.7]
])
FIRST_SOLUTION = np.array([8.4, 4.5, 3.3, 14.3])
THIRD_MATRIX = np.array([
    [3.1, 2.8, 4.9],
    [1.9, 4.1, 2.1],
    [7.5, 3.8, 4.8]
])
SECOND_SOLUTION = np.array([0.2, 2.1, 5.6])


def gram_schmidt_qr(matrix):
    """
    Perform QR decomposition using Gram-Schmidt orthogonalization process.

    :params: 
        matrix: numpy.ndarray - Input matrix to decompose
        
    :returns:
        q_matrix: numpy.ndarray - Orthogonal matrix Q
        r_matrix: numpy.ndarray - Upper triangular matrix R
    """
    num_columns = matrix.shape[1]
    num_rows = matrix.shape[0]
    q_matrix = np.zeros((num_rows, num_columns), dtype=float)
    count = 0

    for column in matrix.T:
        vector_u = np.copy(column)
        for i in range(0, count):
            vector_u = vector_u - np.dot(np.dot(q_matrix[:, i], column), q_matrix[:, i])
        vector_e = vector_u / np.linalg.norm(vector_u)
        q_matrix[:, count] = vector_e
        count += 1

    r_matrix = np.dot(q_matrix.T, matrix)

    return q_matrix, r_matrix


def gram_schmidt_partial(matrix):
    """
    Perform partial QR decomposition using 
    Gram-Schmidt for matrices with 2 columns.
    
    :params: 
        matrix: numpy.ndarray - Input matrix with at least 2 columns
        
    :returns:
        q_matrix: numpy.ndarray - Orthogonal matrix Q (partial)
        r_matrix: numpy.ndarray - Upper triangular matrix R (partial)
    """
    matrix = np.array(matrix, dtype=float)
    num_rows, num_columns = matrix.shape
    q_matrix = np.zeros((num_rows, 2))
    r_matrix = np.zeros((2, num_columns))
    vector_v1 = matrix[:, 0].copy()
    r_matrix[0, 0] = np.linalg.norm(vector_v1)

    if r_matrix[0, 0] > 1e-10:
        q_matrix[:, 0] = vector_v1 / r_matrix[0, 0]
    else:
        q_matrix[:, 0] = vector_v1

    if q_matrix[0, 0] < 0:
        q_matrix[:, 0] = -q_matrix[:, 0]
        r_matrix[0, 0] = -r_matrix[0, 0]
    
    vector_v2 = matrix[:, 1].copy()
    r_matrix[0, 1] = np.dot(q_matrix[:, 0], matrix[:, 1])
    vector_v2 -= r_matrix[0, 1] * q_matrix[:, 0]
    r_matrix[1, 1] = np.linalg.norm(vector_v2)

    if r_matrix[1, 1] > 1e-10:
        q_matrix[:, 1] = vector_v2 / r_matrix[1, 1]
    else:
        q_matrix[:, 1] = vector_v2

    if q_matrix[0, 1] < 0:
        q_matrix[:, 1] = -q_matrix[:, 1]
        r_matrix[0, 1] = -r_matrix[0, 1]
        r_matrix[1, 1] = -r_matrix[1, 1]

    for j in range(2, num_columns):
        r_matrix[0, j] = np.dot(q_matrix[:, 0], matrix[:, j])
        r_matrix[1, j] = np.dot(q_matrix[:, 1], matrix[:, j])
    
    q_matrix = -q_matrix
    r_matrix = -r_matrix

    return q_matrix, r_matrix


def solve_qr(coefficient_matrix, right_side_vector):
    """
    Solve linear system Ax = b using QR decomposition.
    
    :params:
        coefficient_matrix: numpy.ndarray - Coefficient matrix
        right_side_vector: numpy.ndarray - Right-hand side vector
        
    :returns:
        solution: numpy.ndarray - Solution vector
    """
    q_matrix, r_matrix = gram_schmidt_qr(coefficient_matrix)
    b_hat = np.dot(q_matrix.T, right_side_vector)
    n = len(b_hat)
    solution = np.zeros(n)

    for i in range(n-1, -1, -1):
        solution[i] = b_hat[i]
        for j in range(i+1, n):
            solution[i] -= r_matrix[i, j] * solution[j]
        solution[i] /= r_matrix[i, i]

    return solution


def check_diagonal_dominance(matrix):
    """
    Check if a matrix is diagonally dominant.
    
    :params:
        matrix: numpy.ndarray - Square matrix to check
        
    :returns:
        bool - True if matrix is diagonally dominant, False otherwise
    """
    n = len(matrix)

    for i in range(n):
        diagonal = abs(matrix[i][i])
        row_sum = sum(abs(matrix[i][j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False

    return True


def seidel_method(
        coefficient_matrix, 
        right_side_vector, 
        epsilon=1e-3, 
        max_iterations=100):
    """
    Solve linear system using Gauss-Seidel method.
    
    :params:
        coefficient_matrix: numpy.ndarray - Coefficient matrix
        right_side_vector: numpy.ndarray - Right-hand side vector
        epsilon: float - Convergence tolerance (default: 1e-3)
        max_iterations: int - Maximum number of iterations (default: 100)
        
    :returns:
        solution: numpy.ndarray - Solution vector
    """
    n = len(coefficient_matrix)
    solution = np.zeros(n)
    table = PrettyTable()
    table.field_names = ["Iteration"] + [f"x{i+1}" for i in range(n)] + ["Epsilon"]

    for k in range(max_iterations):
        new_solution = np.copy(solution)
        max_error = 0
        for i in range(n):
            sum1 = sum(coefficient_matrix[i][j] * new_solution[j] for j in range(i))
            sum2 = sum(coefficient_matrix[i][j] * solution[j] for j in range(i+1, n))
            new_solution[i] = ((right_side_vector[i] 
                                - sum1 
                                - sum2) 
                                / coefficient_matrix[i][i])
            error = abs(new_solution[i]-solution[i])
            if error > max_error:
                max_error = error
        iteration_number = k + 1
        solution_strings = [f"{val:.6f}" for val in new_solution]
        error_string = f"{max_error:.6f}"
        table.add_row([iteration_number] + solution_strings + [error_string])
        if max_error < epsilon:
            print(table)
            return new_solution
        solution = new_solution

    print(table)

    return solution


def cubic_function(x_value):
    """
    Cubic function: f(x) = -1.38x³ - 5.42x² + 2.57x + 10.95.
    
    :params:
        x_value: float - Input value
        
    :returns:
        float - Function value at x
    """
    return -1.38 * x_value**3 - 5.42 * x_value**2 + 2.57 * x_value + 10.95


def first_derivative(x_value):
    """
    First derivative of the cubic function.
    
    :params:
        x_value: float - Input value
        
    :returns:
        float - Derivative value at x
    """
    return -4.14 * x_value**2 - 10.84 * x_value + 2.57


def second_derivative(x_value):
    """
    Second derivative of the cubic function.
    
    :params:
        x_value: float - Input value
        
    :returns:
        float - Second derivative value at x
    """
    return -8.28 * x_value - 10.84


def bisection_method(
        function, left_boundary, 
        right_boundary, 
        epsilon=1e-3, 
        max_iterations=100):
    """
    Find root of function using bisection method.
    
    :params:
        function: callable - Function to find root of
        left_boundary: float - Left interval boundary
        right_boundary: float - Right interval boundary
        epsilon: float - Convergence tolerance (default: 1e-3)
        max_iterations: int - Maximum number of iterations (default: 100)
        
    :returns:
        tuple - (root, table) where root is the found root or None,
                and table is the iteration history
    """
    table = PrettyTable()
    table.field_names = ["Iteration", "a", "b", "x", "f(a)", "f(b)", "f(x)", "|b-a|"]
    
    if function(left_boundary)*function(right_boundary) > 0:
        return None, "Function has same signs at interval endpoints"

    for i in range(max_iterations):
        x = (left_boundary + right_boundary) / 2
        f_a = function(left_boundary)
        f_b = function(right_boundary)
        f_x = function(x)
        table.add_row([
            i+1,
            f"{left_boundary:.6f}",
            f"{right_boundary:.6f}",
            f"{x:.6f}",
            f"{f_a:.6f}",
            f"{f_b:.6f}",
            f"{f_x:.6f}",
            f"{abs(right_boundary - left_boundary):.6f}"
        ])
        if abs(right_boundary-left_boundary) < epsilon:
            return x, table
        if function(left_boundary)*function(x) < 0:
            right_boundary = x
        else:
            left_boundary = x

    return x, table


def combined_method(
        function, 
        derivative_function, 
        left_boundary, 
        right_boundary, 
        epsilon=1e-5, 
        max_iterations=100):
    """
    Find root using combined chord and tangent method.
    
    :params:
        function: callable - Function to find root of
        derivative_function: callable - Derivative of the function
        left_boundary: float - Left interval boundary
        right_boundary: float - Right interval boundary
        epsilon: float, - Convergence tolerance (default: 1e-5)
        max_iterations: int - Maximum number of iterations (default: 100)
        
    :returns:
        tuple - (root, table) where root is the found root,
                and table is the iteration history
    """
    table = PrettyTable()
    table.field_names = [
        "Iteration", 
        "x_chord", 
        "x_tangent", 
        "f(x_chord)", 
        "f(x_tangent)", 
        "|diff|"
    ]
    x_chord, x_tangent = left_boundary, right_boundary

    for i in range(max_iterations):
        x_chord_new = (x_chord 
                       - function(x_chord) 
                       * (right_boundary - x_chord) 
                       / (function(right_boundary)
                       - function(x_chord)))
        x_tangent_new = x_tangent - function(x_tangent)/derivative_function(x_tangent)
        diff = abs(x_tangent_new - x_chord_new)
        table.add_row([
            i+1,
            f"{x_chord_new:.3f}",
            f"{x_tangent_new:.3f}",
            f"{function(x_chord_new):.3f}",
            f"{function(x_tangent_new):.3f}",
            f"{diff:.3f}"
        ])
        if diff < epsilon:
            return (x_chord_new+x_tangent_new) / 2, table
        x_chord, x_tangent = x_chord_new, x_tangent_new

    return (x_chord+x_tangent) / 2, table


def first_system_equation(x_value, y_value):
    """
    First equation of the system: sin(x) + 2y - 2 = 0.
    
    :params:
        x_value: float - x coordinate
        y_value: float - y coordinate
        
    :returns:
        float - Function value
    """
    return np.sin(x_value) + 2*y_value - 2


def second_system_equation(x_value, y_value):
    """
    Second equation of the system: 2x + cos(y-1) - 0.7 = 0.
    
    :params:
        x_value: float - x coordinate
        y_value: float - y coordinate
        
    :returns:
        float - Function value
    """
    return 2*x_value + np.cos(y_value - 1) - 0.7


def jacobian_matrix(x_value, y_value):
    """
    Compute Jacobian matrix for the system of equations.
    
    :params:
        x_value: float - x coordinate
        y_value: float - y coordinate
        
    :returns:
        numpy.ndarray - 2x2 Jacobian matrix
    """
    df1_dx = np.cos(x_value)
    df1_dy = 2
    df2_dx = 2
    df2_dy = -np.sin(y_value - 1)

    return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])


def newton_system_solver(
        equation1, equation2, jacobian_func, x_initial, 
        y_initial, epsilon=1e-4, max_iterations=100):
    """
    Solve system of equations using Newton's method.
    
    :params:
        equation1: callable - First equation f1(x,y)
        equation2: callable - Second equation f2(x,y)
        jacobian_func: callable - Function to compute Jacobian matrix
        x_initial: float - Initial x guess
        y_initial: float - Initial y guess
        epsilon: float - Convergence tolerance (default: 1e-4)
        max_iterations: int - Maximum number of iterations (default: 100)
        
    :returns:
        tuple - (solution, table, iterations) where solution is (x,y) or None,
                table is iteration history, iterations is number of iterations
    """
    table = PrettyTable()
    table.field_names = ["Iteration", "x", "y", "f1(x,y)", "f2(x,y)", "||delta||"]
    x_current, y_current = x_initial, y_initial

    for i in range(max_iterations):
        f_vector = np.array([
            equation1(x_current, y_current), 
            equation2(x_current, y_current)
        ])
        jacobian = jacobian_func(x_current, y_current)
        determinant_jacobian = np.linalg.det(jacobian)
        
        if abs(determinant_jacobian) < 1e-12:
            return None, table, "Jacobian is singular"
        
        jacobian_dx = np.array([
            [f_vector[0], jacobian[0, 1]], 
            [f_vector[1], jacobian[1, 1]]
        ])
        jacobian_dy = np.array([
            [jacobian[0, 0], f_vector[0]], 
            [jacobian[1, 0], f_vector[1]]
        ])

        delta_x = np.linalg.det(jacobian_dx) / determinant_jacobian
        delta_y = np.linalg.det(jacobian_dy) / determinant_jacobian
        
        x_new = x_current - delta_x
        y_new = y_current - delta_y
        delta_norm = np.sqrt(delta_x**2 + delta_y**2)
        
        table.add_row([
            i+1,
            f"{x_new:.8f}",
            f"{y_new:.8f}",
            f"{equation1(x_new, y_new):.8f}",
            f"{equation2(x_new, y_new):.8f}",
            f"{delta_norm:.8f}"
        ])

        if delta_norm < epsilon:
            return (x_new, y_new), table, i+1
        
        x_current, y_current = x_new, y_new

    return (x_current, y_current), table, max_iterations


def main():
    """
    Main function for calculations.

    :params: None
    
    :returns: None
    """
    np.set_printoptions(precision=4, suppress=True)
    q1_matrix, r1_matrix = gram_schmidt_qr(FIRST_MATRIX)
    print("Matrix A: ", "\n", FIRST_MATRIX)
    print("Matrix Q: ", "\n", q1_matrix, "\n", "Matrix R: ", "\n", r1_matrix)
    print("np.linalg: ", "\n", np.linalg.qr(FIRST_MATRIX))

    q2_matrix, r2_matrix = gram_schmidt_qr(SECOND_MATRIX)
    print("\nMatrix Q:")
    print(q2_matrix)
    print("\nMatrix R:")
    print(r2_matrix)
    qr_matrix = solve_qr(SECOND_MATRIX, FIRST_SOLUTION)
    print(f"\nSystem solution (QR method): x = {qr_matrix}")
    np_solution = np.linalg.solve(SECOND_MATRIX, FIRST_SOLUTION)
    print(f"Numpy solution: x = {np_solution}")

    if not check_diagonal_dominance(THIRD_MATRIX):
        coefficient_matrix = np.array([
            [7.5, 3.8, 4.8],
            [1.9, 4.1, 2.1],
            [3.1, 2.8, 4.9]
        ])
        right_side_vector = np.array([5.6, 2.1, 0.2])
    else:
        coefficient_matrix = THIRD_MATRIX
        right_side_vector = SECOND_SOLUTION

    print("\nZeidel Solution: ")
    solution = seidel_method(coefficient_matrix, right_side_vector)
    print(f"\nSolution: ")

    for i, val in enumerate(solution):
        print(f"x{i+1} = {val:.6f}")

    print("Linalg solve: ", np.linalg.solve(THIRD_MATRIX, SECOND_SOLUTION))

    x_values = np.linspace(-5, 3, 1000)
    y_values = cubic_function(x_values)

    plt.figure(figsize=(12, 8))
    plt.plot(x_values, y_values, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()

    print("\nInterval Analysis")
    analysis_table = PrettyTable()
    analysis_table.field_names = ["x", "f(x)", "f'(x)", "f''(x)", "Sign f(x)"]

    test_points = [-5, -4, -3, -2, -1, 0, 1, 2, 3]

    for point in test_points:
        f_x = cubic_function(point)
        f_prime_x = first_derivative(point)
        f_double_prime_x = second_derivative(point)
        sign = "+" if f_x > 0 else "-" if f_x < 0 else "0"
        analysis_table.add_row([
            point, 
            f"{f_x:.3f}", 
            f"{f_prime_x:.3f}", 
            f"{f_double_prime_x:.3f}", 
            sign
        ])

    print(analysis_table)
    print("\nRoot intervals")
    intervals = []

    for i in range(len(test_points)-1):
        if cubic_function(test_points[i]) * cubic_function(test_points[i+1]) <= 0:
            intervals.append((test_points[i], test_points[i+1]))
            print(f"Root in interval [{test_points[i]}, {test_points[i+1]}]")
            print(f"f({test_points[i]}) = {cubic_function(test_points[i]):.3f},"
                  f"f({test_points[i+1]}) = {cubic_function(test_points[i+1]):.3f}")

    for i, (left_bound, right_bound) in enumerate(intervals):
        print(f"\nSolution for root in interval [{left_bound}, {right_bound}]")
        print("\nConvergence conditions check:")
        print(f"f({left_bound}) = {cubic_function(left_bound):.3f}")
        print(f"f({right_bound}) = {cubic_function(right_bound):.3f}")
        print(f"f({left_bound}) \
              * f({right_bound}) \
              = {cubic_function(left_bound) \
              * cubic_function(right_bound):.3f}"
        )

        if cubic_function(left_bound)*cubic_function(right_bound) < 0:
            print("Condition f(a)*f(b) < 0 is satisfied")
        else:
            print("Condition f(a)*f(b) < 0 is not satisfied")

        print(f"\nBisection method")
        root_bisection, table_bisection = bisection_method(
            cubic_function, 
            left_bound, 
            right_bound, 
            epsilon=1e-3
        )
        
        if root_bisection is not None:
            print(table_bisection)
            print(f"Found root: x = {root_bisection:.3f}")
            print(f"Check: f({root_bisection:.3f}) = {cubic_function(root_bisection):.3f}")
        else:
            print(table_bisection)

        print(f"\nCombined method")
        root_combined, table_combined = combined_method(
            cubic_function,
            first_derivative,
            left_bound,
            right_bound,
            epsilon=1e-5
        )
        print(table_combined)
        print(f"Found root: x = {root_combined:.3f}")
        print(f"Check: f({root_combined:.3f}) = {cubic_function(root_combined):.3f}")

    print(f"\nNumpy Solution")
    coefficients = [-1.38, -5.42, 2.57, 10.95]
    roots_numpy = np.roots(coefficients)
    real_roots = roots_numpy[np.isreal(roots_numpy)].real

    for i, root in enumerate(real_roots):
        if -5 <= root <= 3:
            print(f"Root {i+1}: x = {root:.3f}")
            print(f"Check: f({root:.3f}) = {cubic_function(root):.3f}")

    print("Function analysis")
    print("System of equations:")
    print("f1(x,y) = sin(x) + 2y - 2 = 0")
    print("f2(x,y) = 2x + cos(y-1) - 0.7 = 0")

    x_range = np.linspace(-2, 2, 100)
    y_range = np.linspace(-2, 2, 100)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)

    z1_values = first_system_equation(X_grid, Y_grid)
    z2_values = second_system_equation(X_grid, Y_grid)

    plt.figure(figsize=(10, 8))
    contour1 = plt.contour(
        X_grid, 
        Y_grid, 
        z1_values, 
        levels=[0], 
        colors='red', 
        linewidths=2
    )
    contour2 = plt.contour(
        X_grid, 
        Y_grid, 
        z2_values, 
        levels=[0], 
        colors='blue', 
        linewidths=2
    )
    plt.grid(True, alpha=0.3)
    plt.show()

    x_initial, y_initial = 0.5, 0.8
    print("\nJacobian matrix")
    jacobian = jacobian_matrix(x_initial, y_initial)
    print("J(x,y) =")
    print(f"\nAt initial point (x0,y0) = ({x_initial},{y_initial}):")
    print(f"J({x_initial},{y_initial}) =")
    print(f"[ {jacobian[0, 0]:.6f}  {jacobian[0, 1]:.6f} ]")
    print(f"[ {jacobian[1, 0]:.6f}  {jacobian[1, 1]:.6f} ]")

    print("\nLinear system")
    print("System to solve for delta_x, delta_y:")
    print(f"[ {jacobian[0, 0]:.6f}  {jacobian[0, 1]:.6f} ] "
          f"[delta_x] = [{-first_system_equation(x_initial, y_initial):.6f}]")
    print(f"[ {jacobian[1, 0]:.6f}  {jacobian[1, 1]:.6f} ] "
          f"[delta_y] = [{-second_system_equation(x_initial, y_initial):.6f}]")

    print("\nNewton's method solution")
    solution, table, iterations = newton_system_solver(
        first_system_equation,
        second_system_equation,
        jacobian_matrix,
        x_initial,
        y_initial
    )
    print(table)


if __name__ == "__main__":
    main()