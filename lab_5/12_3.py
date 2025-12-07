import math
import matplotlib.pyplot as plt
import numpy as np 
import scipy as sp
from prettytable import PrettyTable
from scipy.optimize import fsolve

first_matrix = np.random.randint(-8, 9, (3, 3))
print("Matrix A: ", "\n", first_matrix)

def gram_schmidt_qr(A):
    """
    Perform QR decomposition using the Gram-Schmidt orthogonalization process.
    
    Parameters:
    A: numpy.ndarray
        Input matrix to decompose
        
    Returns:
    Q: numpy.ndarray
        Orthogonal matrix Q
    R: numpy.ndarray
        Upper triangular matrix R
    """
    n = A.shape[1]
    m = A.shape[0]
    Q = np.zeros((m, n), dtype=float)
    cnt = 0

    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u = u - np.dot(np.dot(Q[:, i], a), Q[:, i]) 
        e = u/np.linalg.norm(u)
        Q[:, cnt] = e
        cnt += 1

    R = np.dot(Q.T, A)

    return Q, R


def gram_schmidt_partial(A):
    """
    Perform partial QR decomposition using 
    Gram-Schmidt for matrices with 2 columns.
    
    Parameters:
    A: numpy.ndarray
        Input matrix with at least 2 columns
        
    Returns:
    Q: numpy.ndarray
        Orthogonal matrix Q (partial)
    R: numpy.ndarray
        Upper triangular matrix R (partial)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, 2))
    R = np.zeros((2, n))
    v1 = A[:, 0].copy()
    R[0, 0] = np.linalg.norm(v1)

    if R[0, 0] > 1e-10:
        Q[:, 0] = v1 / R[0, 0]
    else: 
        Q[:, 0] = v1

    if Q[0, 0] < 0:
        Q[:, 0] = -Q[:, 0]
        R[0, 0] = -R[0, 0]
    v2 = A[:, 1].copy()
    R[0, 1] = np.dot(Q[:, 0], A[:, 1])
    v2 -= R[0, 1] * Q[:, 0]
    R[1, 1] = np.linalg.norm(v2)

    if R[1, 1] > 1e-10:
        Q[:, 1] = v2/R[1, 1]
    else: 
        Q[:, 1] = v2

    if Q[0, 1] < 0:
        Q[:, 1] = -Q[:, 1]
        R[0, 1] = -R[ 0, 1 ]
        R[1, 1]= -R[ 1, 1 ]

    for j in range(2, n):
        R[0, j] = np.dot(Q[:, 0], A[:, j])
        R[1, j] = np.dot(Q[:, 1], A[:, j])
    Q = -Q
    R = -R

    return Q, R


np.set_printoptions(precision=4, suppress=True)
q1_matrix, r1_matrix = gram_schmidt_qr(first_matrix)
print("Matrix Q: ", "\n", q1_matrix, "\n", "Matrix R: ", "\n", r1_matrix)
print("np.linalg: ", "\n", np.linalg.qr(first_matrix))

second_matrix = np.array([
    [8.2, 3.2, 14.2, 14.8], 
    [5.6, 12, 15, 6.4],
    [5.7, 3.6, 12.4, 2.3],
    [6.8, 13.2, 6.3, 8.7]
])
first_solution = np.array([8.4, 4.5, 3.3, 14.3])

def solve_qr(A, b):
    """
    Solve linear system Ax = b using QR decomposition.
    
    Parameters:
    A: numpy.ndarray
        Coefficient matrix
    b: numpy.ndarray
        Right-hand side vector
        
    Returns:
    x: numpy.ndarray
        Solution vector
    """
    Q, R = gram_schmidt_qr(A)
    b_hat = np.dot(Q.T, b)
    n = len(b_hat)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = b_hat[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    
    return x


q2_matrix, r2_matrix = gram_schmidt_qr(second_matrix)
print("\nMatrix Q:")
print(q2_matrix)
print("\nMatrix R:")
print(r2_matrix)
qr_matrix = solve_qr(second_matrix, first_solution)
print(f"\nSystem solution (QR method): x = {qr_matrix}")
np_solution = np.linalg.solve(second_matrix, first_solution)
print(f"Numpy solution: x = {np_solution}")

def check_diagonal_dominance(A):
    """
    Check if a matrix is diagonally dominant.
    
    Parameters:
    A: numpy.ndarray
        Square matrix to check
        
    Returns:
    bool
        True if matrix is diagonally dominant, False otherwise
    """
    n = len(A)

    for i in range(n):
        diagonal = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False
        
    return True


def seidel_method(A, b, eps=1e-3, max_iter=100):
    """
    Solve linear system using Gauss-Seidel method.
    
    Parameters:
    A: numpy.ndarray
        Coefficient matrix
    b: numpy.ndarray
        Right-hand side vector
    eps: float, optional
        Convergence tolerance (default: 1e-3)
    max_iter: int, optional
        Maximum number of iterations (default: 100)
        
    Returns:
    x: numpy.ndarray
        Solution vector
    """
    n = len(A)
    x = np.zeros(n)
    table = PrettyTable()
    table.field_names = ["Iteration"] + [f"x{i + 1}" for i in range(n)] + ["Epsilon"]
    
    for k in range(max_iter):
        x_new = np.copy(x)
        max_error = 0
        for i in range(n):
            s1 = sum(A[i][j]*x_new[j] for j in range(i))
            s2 = sum(A[i][j]*x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2)/A[i][i]
            error = abs(x_new[i] - x[i])
            if error > max_error:
                max_error = error
        table.add_row([k+1] + [f"{val:.6f}" for val in x_new] + [f"{max_error:.6f}"])
        if max_error < eps:
            print(table)
            return x_new
        x = x_new

    print(table)

    return x


third_matrix = np.array([
    [3.1, 2.8, 4.9], 
    [1.9, 4.1, 2.1],
    [7.5, 3.8, 4.8]
])
second_solution = np.array([0.2, 2.1, 5.6])

if not check_diagonal_dominance(third_matrix):
    A = np.array([
        [7.5, 3.8, 4.8], 
        [1.9, 4.1, 2.1],
        [3.1, 2.8, 4.9]
    ])
    b = np.array([5.6, 2.1, 0.2])
else:
    A = third_matrix
    b = second_solution

print("\nZeidel Solution: ")
solution = seidel_method(A, b)
print(f"\nSolution: ")

for i, val in enumerate(solution):
    print(f"x{i+1} = {val:.6f}")

print("Linalg solve: ", np.linalg.solve(third_matrix, second_solution))

def cubic_function(x):
    """
    Cubic function: f(x) = -1.38x³ - 5.42x² + 2.57x + 10.95.
    
    Parameters:
    x: float
        Input value
        
    Returns:
    float
        Function value at x
    """
    return -1.38*x**3 - 5.42*x**2 + 2.57*x + 10.95


def f_prime(x):
    """
    First derivative of the cubic function.
    
    Parameters:
    x: float
        Input value
        
    Returns:
    float
        Derivative value at x
    """
    return -4.14*x**2 - 10.84*x + 2.57


def f_double_prime(x):
    """
    Second derivative of the cubic function.
    
    Parameters:
    x: float
        Input value
        
    Returns:
    float
        Second derivative value at x
    """
    return -8.28*x - 10.84


def bisection_method(f, a, b, eps=1e-3, max_iter=100):
    """
    Find root of function using bisection method.
    
    Parameters:
    f: callable
        Function to find root of
    a: float
        Left interval boundary
    b: float
        Right interval boundary
    eps: float, optional
        Convergence tolerance (default: 1e-3)
    max_iter: int, optional
        Maximum number of iterations (default: 100)
        
    Returns:
    tuple
        (root, table) where root is the found root or None,
        and table is the iteration history
    """
    table = PrettyTable()
    table.field_names = ["Iteration", "a", "b", "x", "f(a)", "f(b)", "f(x)", "|b-a|"]
    if f(a)*f(b) > 0:
        return None, "Function has same signs at interval endpoints"
    
    for i in range(max_iter):
        x = (a+b)/2
        fa, fb, fx = f(a), f(b), f(x)
        table.add_row([
            i+1, 
            f"{a:.6f}", 
            f"{b:.6f}",   
            f"{x:.6f}", 
            f"{fa:.6f}", 
            f"{fb:.6f}", 
            f"{fx:.6f}", 
            f"{abs(b-a):.6f}"
        ])
        if abs(b-a) < eps:
            return x, table
        if f(a)*f(x) < 0:  
            b = x
        else: 
            a = x

    return x, table


def combined_method(f, f_prime, a, b, eps=1e-5, max_iter=100):
    """
    Find root using combined chord and tangent method.
    
    Parameters:
    f: callable
        Function to find root of
    f_prime: callable
        Derivative of the function
    a: float
        Left interval boundary
    b: float
        Right interval boundary
    eps: float, optional
        Convergence tolerance (default: 1e-5)
    max_iter: int, optional
        Maximum number of iterations (default: 100)
        
    Returns:
    tuple
        (root, table) where root is the found root,
        and table is the iteration history
    """
    table = PrettyTable()
    table.field_names = ["Iteration", "x_chord", "x_tangent", "f(x_chord)", "f(x_tangent)", "|diff|"]
    x_chord, x_tangent = a, b

    for i in range(max_iter):
        x_chord_new = x_chord - f(x_chord)*(b-x_chord) / (f(b)-f(x_chord))
        x_tangent_new = x_tangent - f(x_tangent)/f_prime(x_tangent)
        diff = abs(x_tangent_new-x_chord_new)
        table.add_row([
            i+1, 
            f"{x_chord_new:.3f}", 
            f"{x_tangent_new:.3f}",  
            f"{f(x_chord_new):.3f}", 
            f"{f(x_tangent_new):.3f}", 
            f"{diff:.3f}"
        ])
        if diff < eps:
            return (x_chord_new+x_tangent_new)/2, table
        x_chord, x_tangent = x_chord_new, x_tangent_new

    return (x_chord+x_tangent)/2, table


x = np.linspace(-5, 3, 1000)
y = cubic_function(x)

plt.figure(figsize=(12, 8))
plt.plot(x, y, 'b-', linewidth=2)
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
    fx = cubic_function(point)
    fpx = f_prime(point)
    fdpx = f_double_prime(point)
    sign = "+" if fx > 0 else "-" if fx < 0 else "0"
    analysis_table.add_row([point, f"{fx:.3f}", f"{fpx:.3f}", f"{fdpx:.3f}", sign])

print(analysis_table)
print("\nRoot intervals")
intervals = []

for i in range(len(test_points)-1):
    if cubic_function(test_points[i])*cubic_function(test_points[i+1]) <= 0:
        intervals.append((test_points[i], test_points[i+1]))
        print(f"Root in interval [{test_points[i]}, {test_points[i+1]}]")
        print(f"f({test_points[i]}) = {cubic_function(test_points[i]):.3f}, 
              f({test_points[i+1]}) = {cubic_function(test_points[i+1]):.3f}")

for i, (a, b) in enumerate(intervals):
    print(f"\nSolution for root in interval [{a}, {b}]")
    print("\nConvergence conditions check:")
    print(f"f({a}) = {cubic_function(a):.3f}")
    print(f"f({b}) = {cubic_function(b):.3f}")
    print(f"f({a}) * f({b}) = {cubic_function(a) * cubic_function(b):.3f}")
    
    if cubic_function(a)*cubic_function(b) < 0: 
        print("Condition f(a)*f(b) < 0 is satisfied")
    else:  
        print("Condition f(a)*f(b) < 0 is not satisfied")
    
    print(f"\nBisection method")
    root_bisection, table_bisection = bisection_method(cubic_function, a, b, eps=1e-3)
    if root_bisection is not None:
        print(table_bisection)
        print(f"Found root: x = {root_bisection:.3f}")
        print(f"Check: f({root_bisection:.3f}) = {cubic_function(root_bisection):.3f}")
    else: 
        print(table_bisection)
    
    print(f"\nCombined method")
    root_combined, table_combined = combined_method(
        cubic_function, 
        f_prime, a, 
        b, eps=1e-5
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

def first_equation(x, y):
    """
    First equation of the system: sin(x) + 2y - 2 = 0.
    
    Parameters:
    x: float
        x coordinate
    y: float
        y coordinate
        
    Returns:
    float
        Function value
    """
    return np.sin(x) + 2*y - 2


def second_equation(x, y):
    """
    Second equation of the system: 2x + cos(y-1) - 0.7 = 0.
    
    Parameters:
    x: float
        x coordinate
    y: float
        y coordinate
        
    Returns:
    float
        Function value
    """
    return 2*x + np.cos(y-1) - 0.7


def jacobian(x, y):
    """
    Compute Jacobian matrix for the system of equations.
    
    Parameters:
    x: float
        x coordinate
    y: float
        y coordinate
        
    Returns:
    numpy.ndarray
        2x2 Jacobian matrix
    """
    df1_dx = np.cos(x)
    df1_dy = 2
    df2_dx = 2
    df2_dy = -np.sin(y-1)

    return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])


def newton_system(f1, f2, jacobian, x0, y0, eps=1e-4, max_iter=100):
    """
    Solve system of equations using Newton's method.
    
    Parameters:
    f1: callable
        First equation f1(x,y)
    f2: callable
        Second equation f2(x,y)
    jacobian_func: callable
        Function to compute Jacobian matrix
    x0: float
        Initial x guess
    y0: float
        Initial y guess
    eps: float, optional
        Convergence tolerance (default: 1e-4)
    max_iter: int, optional
        Maximum number of iterations (default: 100)
        
    Returns:
    tuple
        (solution, table, iterations) where solution is (x,y) or None,
        table is iteration history, iterations is number of iterations
    """
    table = PrettyTable()
    table.field_names = ["Iteration", "x", "y", "f1(x,y)", "f2(x,y)", "||Δ||"]
    x, y = x0, y0
    
    for i in range(max_iter):
        F = np.array([f1(x, y), f2(x, y)])
        J = jacobian(x, y)
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            return None, table, "Jacobian is singular"
        J_dx = np.array([[F[0], J[0,1]], [F[1], J[1,1]]])
        J_dy = np.array([[J[0,0], F[0]], [J[1,0], F[1]]])
        
        dx = np.linalg.det(J_dx)/det_J
        dy = np.linalg.det(J_dy)/det_J
        x_new = x-dx
        y_new = y-dy
        delta_norm = np.sqrt(dx**2 + dy**2)
        table.add_row([
            i+1, 
            f"{x_new:.8f}", 
            f"{y_new:.8f}", 
              f"{f1(x_new, y_new):.8f}", 
              f"{f2(x_new, y_new):.8f}",  
              f"{delta_norm:.8f}"
        ])
        
        if delta_norm < eps:
            return (x_new, y_new), table, i+1
        x, y = x_new, y_new

    return (x, y), table, max_iter


print("Function analysis")
print("System of equations:")
print("f1(x,y) = sin(x) + 2y - 2 = 0")
print("f2(x,y) = 2x + cos(y-1) - 0.7 = 0")

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

z1 = first_equation(X, Y)
z2 = second_equation(X, Y)

plt.figure(figsize=(10, 8))
contour1=plt.contour(X, Y, z1, levels=[0], colors='red', linewidths=2)
contour2=plt.contour(X, Y, z2, levels=[0], colors='blue', linewidths=2)
plt.grid(True, alpha=0.3)
plt.show()

x0, y0 = 0.5, 0.8
print("\nJacobian matrix")
jacobian_matrix = jacobian(x0, y0)
print("J(x,y) =")
print(f"[ cos(x)      2     ]")
print(f"[   2     -sin(y-1) ]")
print(f"\nAt initial point (x0,y0) = ({x0},{y0}):")
print(f"J({x0},{y0}) =")
print(f"[ {jacobian_matrix[0,0]:.6f}  {jacobian_matrix[0,1]:.6f} ]")
print(f"[ {jacobian_matrix[1,0]:.6f}  {jacobian_matrix[1,1]:.6f} ]")

print("\nLinear system")
print("System to solve for Δx, Δy:")
print(f"[ {jacobian_matrix[0,0]:.6f}  {jacobian_matrix[0,1]:.6f} ] 
      [Δx]   [{-first_equation(x0,y0):.6f}]")
print(f"[ {jacobian_matrix[1,0]:.6f}  {jacobian_matrix[1,1]:.6f} ] 
      [Δy] = [{-second_equation(x0,y0):.6f}]")
print("\nNewton's method solution")
solution, table, iterations = newton_system(
    first_equation, 
    second_equation,
    jacobian, x0, y0
)
print(table)