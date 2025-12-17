import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sys
from prettytable import PrettyTable
from scipy.optimize import fsolve
from config_manager import get_config


class NumericalMethods:
    def __init__(self, config=None):
        """
        Initialize numerical methods.
        
        :params:
            config: NumericalMethodsConfig - Configuration object
        """
        self.config = config or get_config()
        self.results = {}
        np.set_printoptions(precision=4, suppress=True)


    def run_qr_decomposition(self):
        """
        Perform QR decomposition using Gram-Schmidt method.
        """
        min_val = self.config.config.matrix.range_min
        max_val = self.config.config.matrix.range_max
        size = self.config.config.matrix.size
        
        matrix_a = np.random.randint(min_val, max_val + 1, (size, size))
        
        print("Matrix A:")
        print(matrix_a)
        
        q_matrix, r_matrix = self.gram_schmidt_qr(matrix_a)
        
        print("\nMatrix Q:")
        print(q_matrix)
        print("\nMatrix R:")
        print(r_matrix)
        
        if self.config.config.validation.compare_with_numpy:
            q_np, r_np = np.linalg.qr(matrix_a)
            print("\nNumPy QR (for comparison):")
            print("Q (NumPy):")
            print(q_np)
            print("R (NumPy):")
            print(r_np)
        
        return q_matrix, r_matrix


    def gram_schmidt_qr(self, matrix):
        """
        Perform QR decomposition using Gram-Schmidt orthogonalization.
        
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
                vector_u = vector_u - np.dot(np.dot(q_matrix[:, i], column), 
                                           q_matrix[:, i])
            vector_e = vector_u / np.linalg.norm(vector_u)
            q_matrix[:, count] = vector_e
            count += 1

        r_matrix = np.dot(q_matrix.T, matrix)

        return q_matrix, r_matrix


    def solve_linear_system_qr(self):
        """
        Solve linear system using QR decomposition.
        """
        mat = np.array(self.config.config.qr_test.mat)
        vec = np.array(self.config.config.qr_test.vec)
        
        rows, cols = mat.shape[0], mat.shape[1]
        print(f"Coefficient matrix: {rows}x{cols}")
        print(mat)
        print(f"\nRight-hand side vector: {vec}")
        
        solution = self.solve_qr(mat, vec)
        
        print(f"\nSolution (QR method): x = {solution}")
        
        return solution
    

    def solve_qr(self, coefficient_matrix, right_side_vector):
        """
        Solve linear system Ax = b using QR decomposition.
        
        :params:
            coefficient_matrix: numpy.ndarray - Coefficient matrix
            right_side_vector: numpy.ndarray - Right-hand side vector
            
        :returns:
            solution: numpy.ndarray - Solution vector
        """
        q_matrix, r_matrix = self.gram_schmidt_qr(coefficient_matrix)
        b_hat = np.dot(q_matrix.T, right_side_vector)
        n = len(b_hat)
        solution = np.zeros(n)

        for i in range(n-1, -1, -1):
            solution[i] = b_hat[i]
            for j in range(i+1, n):
                solution[i] -= r_matrix[i, j] * solution[j]
            solution[i] /= r_matrix[i, i]

        return solution


    def solve_seidel(self):
        """
        Solve linear system using Gauss-Seidel method.
        """
        mat = np.array(self.config.config.seidel.mat)
        vec = np.array(self.config.config.seidel.vec)
        
        print("Matrix A:")
        print(mat)
        print(f"\nVector b: {vec}")
        
        solution = self.seidel_method(mat, vec)
        
        print(f"\nSolution: {solution}")
        
        return solution


    def check_diagonal_dominance(self, matrix):
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


    def seidel_method(self, coefficient_matrix, right_side_vector, 
                                   epsilon=1e-3, max_iterations=100):
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
                sum1 = sum(coefficient_matrix[i][j] * new_solution[j] 
                          for j in range(i))
                sum2 = sum(coefficient_matrix[i][j] * solution[j] 
                          for j in range(i+1, n))
                new_solution[i] = ((right_side_vector[i] - sum1 - sum2) 
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
    

    def solve_nonlinear_equation(self):
        """
        Solve nonlinear equation using bisection and combined methods.
        """
        coeffs = self.config.config.nonlinear_equations.function_coeffs
        eq_str = (f"f(x) = {coeffs[0]:.2f}x³ "
                  f"{coeffs[1]:+.2f}x² "
                  f"{coeffs[2]:+.2f}x "
                  f"{coeffs[3]:+.2f}")
        print(f"Function: {eq_str}")
        
        test_points = self.config.config.nonlinear_equations.root_finding.test_points
        
        print("\nInterval analysis:")
        intervals = []
        
        for i in range(len(test_points) - 1):
            left = test_points[i]
            right = test_points[i + 1]
            left_val = self.cubic_function(left)
            right_val = self.cubic_function(right)
            
            if left_val * right_val <= 0:
                intervals.append((left, right))
                print(f"Root in interval [{left}, {right}]")
        
        return intervals


    def cubic_function(self, x_value):
        """
        Cubic function from configuration.
        
        :params:
            x_value: float - Input value
            
        :returns:
            float - Function value at x
        """
        coeffs = self.config.config.nonlinear_equations.function_coeffs
        return (coeffs[0] * x_value**3 
                    + coeffs[1] * x_value**2 
                    + coeffs[2] * x_value 
                    + coeffs[3])


    def first_derivative(self, x_value):
        """
        First derivative from configuration.
        
        :params:
            x_value: float - Input value
            
        :returns:
            float - Derivative value at x
        """
        coeffs = self.config.config.nonlinear_equations.first_der_coeffs
        return (coeffs[0] * x_value**2 
                    + coeffs[1] * x_value 
                    + coeffs[2])


    def second_derivative(self, x_value):
        """
        Second derivative from configuration.
        
        :params:
            x_value: float - Input value
            
        :returns:
            float - Second derivative value at x
        """
        coeffs = self.config.config.nonlinear_equations.second_der_coeffs
        return coeffs[0] * x_value + coeffs[1]


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


    def bisection_method(self, function, left_boundary, right_boundary, 
                        epsilon=1e-3, max_iterations=100):
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
        table.field_names = ["Iteration", "a", "b", "x", "f(a)", "f(b)", 
                           "f(x)", "|b-a|"]
        
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


    def combined_method(self, function, derivative_function, left_boundary, 
                       right_boundary, epsilon=1e-5, max_iterations=100):
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
            x_chord_new = (x_chord - function(x_chord) 
                          * (right_boundary - x_chord) 
                          / (function(right_boundary) - function(x_chord)))
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


    def solve_nonlinear_system(self):
        """
        Solve nonlinear system using Newton's method.
        """
        eq1 = self.config.config.nonlinear_systems.newton.eq1
        eq2 = self.config.config.nonlinear_systems.newton.eq2
        
        print("System of equations:")
        print(f"  f1(x,y) = {eq1} = 0")
        print(f"  f2(x,y) = {eq2} = 0")
        
        newton_params = self.config.get_method_params("newton_system")
        x_initial, y_initial = newton_params.get("initial_guess", [0.5, 0.8])
        
        print(f"\nInitial guess: x0 = {x_initial}, y0 = {y_initial}")
        
        solution, table, iterations = self.newton_system_solver(
            self.first_system_equation,
            self.second_system_equation,
            self.jacobian_matrix,
            x_initial,
            y_initial
        )
        
        print(table)
        
        if solution:
            print(f"\nSolution: x = {solution[0]:.6f}, "
                  f"y = {solution[1]:.6f}")
        
        return solution


    def first_system_equation(self, x_value, y_value):
        """
        First equation of the system: sin(x) + 2y - 2 = 0.
        
        :params:
            x_value: float - x coordinate
            y_value: float - y coordinate
            
        :returns:
            float - Function value
        """
        return np.sin(x_value) + 2*y_value - 2


    def second_system_equation(self, x_value, y_value):
        """
        Second equation of the system: 2x + cos(y-1) - 0.7 = 0.
        
        :params:
            x_value: float - x coordinate
            y_value: float - y coordinate
            
        :returns:
            float - Function value
        """
        return 2*x_value + np.cos(y_value - 1) - 0.7


    def jacobian_matrix(self, x_value, y_value):
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


    def newton_system_solver(self, equation1, equation2, jacobian_func, 
                            x_initial, y_initial, epsilon=1e-4, 
                            max_iterations=100):
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
    

    def plot_function(self):
        """
        Plot cubic function from configuration.
        """
        coeffs = self.config.config.nonlinear_equations.function_coeffs
        config = self.config.config.nonlinear_equations.root_finding
        
        x_min, x_max = config.plot_interval
        num_points = config.plot_points
        
        x_vals = np.linspace(x_min, x_max, num_points)
        y_vals = self.cubic_function(x_vals)

        plt.figure(figsize=(12, 8))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        
        title = (f"Cubic Function: "
                 f"f(x) = {coeffs[0]:.2f}x³ "
                 f"{coeffs[1]:+.2f}x² "
                 f"{coeffs[2]:+.2f}x "
                 f"{coeffs[3]:+.2f}")
        plt.title(title)
        
        if self.config.should_show_output("plot"):
            plt.show()


    def plot_system(self):
        """
        Plot system of equations.
        """
        x_range = np.linspace(-2, 2, 100)
        y_range = np.linspace(-2, 2, 100)
        X_grid, Y_grid = np.meshgrid(x_range, y_range)

        z1_values = self.first_system_equation(X_grid, Y_grid)
        z2_values = self.second_system_equation(X_grid, Y_grid)

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
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('System of Equations')
        plt.show()


    def run_all_methods(self):
        """
        Run all numerical methods.
        """
        print(f"Environment: {self.config.env}")
        
        print("\n1. QR Decomposition:")
        self.run_qr_decomposition()
        
        print("\n2. Linear System Solution (QR):")
        self.solve_linear_system_qr()
        
        print("\n3. Seidel Method:")
        self.solve_seidel()
        
        print("\n4. Nonlinear Equation Solving:")
        self.solve_nonlinear_equation()
        
        print("\n5. Nonlinear System Solving:")
        self.solve_nonlinear_system()
        
        if self.config.should_show_output("plot"):
            print("\n6. Plotting Functions:")
            self.plot_function()
            self.plot_system()
        
        print("\nAll methods completed successfully.")


def main():
    """
    Main function for numerical methods.
    """
    parser = argparse.ArgumentParser(
        description="Numerical Methods Laboratory Work",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--env",
        choices=["dev", "prod"],
        default=None,
        help="Runtime environment (overrides NUM_METHODS_ENV)"
    )
    
    parser.add_argument(
        "--method",
        choices=["qr", "seidel", "nonlinear", "system", "all"],
        default="all",
        help="Run specific method (default: all)"
    )
    
    parser.add_argument(
        "--config-dir",
        default="config",
        help="Path to configuration directory"
    )
    
    args = parser.parse_args()
    
    try:
        config = get_config(env=args.env)
        lab = NumericalMethods(config)
        
        match args.method:
            case "all":
                lab.run_all_methods()
            case "qr":
                lab.run_qr_decomposition()
            case "seidel":
                lab.solve_seidel()
            case "nonlinear":
                lab.solve_nonlinear_equation()
            case "system":
                lab.solve_nonlinear_system()
            case _:
                print(f"Unknown method: {args.method}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
