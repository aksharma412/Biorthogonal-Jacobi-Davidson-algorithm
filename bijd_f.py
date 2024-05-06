import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score 

def BiJD_all_eigenvalues(A, m, V, W, tolerance=1e-6, max_iterations=100):
    n = A.shape[0]
    j = len(V[0])  # Initial number of basis vectors

    # Initialize K, L, and H
    K = A @ V
    L = np.conj(A.T) @ W
    H = np.conj(W.T) @ K

    eigenvalues = []  # List to store eigenvalues
    residuals = []  # List to store residuals for convergence plot
    left = []
    right = []

    iteration = 0
    while iteration < max_iterations:
        # Step 1: Compute right (gi) and left (fi) eigenvectors of H
        _, g = np.linalg.eigh(H)
        f = np.linalg.inv(W) @ g

        # Step 2: Target the Ritz triplet (f, g, λ) with Ritz value λ closest to σ
        x = V @ g
        y = W @ f
        rr = K @ g - x * np.diag(x.T @ K @ x)
        rl = L @ f - y * np.diag(y.T @ L @ y)

        # Step 3: Run p steps of BCG simultaneously on the two correction equations
        p = 5  # Number of BCG steps
        for _ in range(p):
        
            i_minus_xy_star = np.eye(n) - x @ np.conj(y.T)
            i_minus_yx_star = np.eye(n) - y @ np.conj(x.T)

            # Solve for delta_r using the first correction equation
            delta_r = np.linalg.solve(i_minus_xy_star @ (A - np.eye(n) @ np.diag(x.T @ K @ x)) @ i_minus_xy_star, rr)

            # Solve for delta_l using the second correction equation
            delta_l = np.linalg.solve(i_minus_yx_star @ (np.conj(A.T) - np.eye(n) @ np.diag(y.T @ L @ y)) @ i_minus_yx_star, rl)

            # Update residuals
            rr -= (A - np.diag(x.T @ A @ x)) @ (i_minus_xy_star @ delta_r)
            rl -= (np.conj(A.T) - np.diag(y.T @ np.conj(A.T) @ y)) @ (i_minus_yx_star @ delta_l)
            left.append(np.linalg.norm(delta_l))
            right.append(np.linalg.norm(delta_r))
            

        # Step 4: Update V and W
        V = np.hstack((V, delta_r))
        W = np.hstack((W, delta_l))

        # Step 5: Orthonormalize the new basis vectors using Modified Gram-Schmidt
        V, _ = np.linalg.qr(V, mode='reduced')
        W, _ = np.linalg.qr(W, mode='reduced')

        # Step 6: Update j, K, L, and H
        j += 1
        K = A @ V
        L = np.conj(A.T) @ W
        H = np.conj(W.T) @ K

        # Step 7: Store eigenvalues and residuals
        eigenvalues.append(np.diag(x.T @ A @ x))
        residuals.append((np.linalg.norm(rr), np.linalg.norm(rl)))
        iteration += 1

        # Step 8: Check if convergence is achieved
        if all(np.linalg.norm(rr) < tolerance and np.linalg.norm(rl) < tolerance for rr, rl in residuals):
            break

    print(iteration)

    return eigenvalues, left, right

# Function to generate the heat map of matrix A
def plot_heatmap(A):
    plt.figure(figsize=(8, 6))
    sns.heatmap(A, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
    plt.title('Heatmap of Matrix A')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()

# Function to plot scatter plot of eigenvalue estimates
def plot_scatter(eigenvalues, true_eigenvalues=None):
    plt.figure(figsize=(8, 6))
    if true_eigenvalues is not None:
        if len(true_eigenvalues) > 1:
            plt.scatter(true_eigenvalues, np.arange(len(true_eigenvalues)), color='blue', label='Estimated True Eigenvalues')
            plt.scatter(eigenvalues, np.arange(len(eigenvalues)), color='red', label='Predicted Eigenvalues')
            plt.annotate("r-squared = {:.3f}".format(r2_score(eigenvalues, true_eigenvalues)), (5,5))
            #plt.set_title('R2: ' + str(r2_score(eigenvalues, true_eigenvalues)))
            plt.plot(true_eigenvalues, true_eigenvalues, linestyle='--', color='green', label='Ideal')
            plt.plot(true_eigenvalues, true_eigenvalues, linestyle='--', color='yellow', label='Real')
        else:
            plt.scatter(range(1, len(eigenvalues) + 1), eigenvalues, color='blue', label='Eigenvalue Estimates')
    else:
        plt.scatter(range(1, len(eigenvalues) + 1), eigenvalues, color='blue', label='Eigenvalue Estimates')
    plt.title('Scatter Plot of Eigenvalue Estimates')
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
# Function to plot convergence of residuals
def plot_convergence(right, left):
    plt.figure(figsize=(8, 6))
    #right = np.array(right)
    #left = np.array(left)
    iterations = np.arange(len(left))
    plt.plot(iterations, right, marker='o', linestyle='-', color='blue', label='Right Residual')
    plt.plot(iterations, left, marker='o', linestyle='-', color='red', label='Left Residual')
    plt.title('Convergence Plot of Residuals')
    plt.xlabel('Iteration')
    plt.ylabel('Residual Norm')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot convergence of approximate eigenvalue sigma over iterations
def plot_sigma_convergence(sigma_values):
    plt.figure(figsize=(8, 6))
    iterations = np.arange(len(sigma_values))
    plt.plot(iterations, sigma_values, marker='o', linestyle='-', color='green')
    plt.title('Convergence Plot of Approximate Eigenvalue Sigma')
    plt.xlabel('Iteration')
    plt.ylabel('Approximate Eigenvalue Sigma')
    plt.grid(True)
    plt.show()
# Test case
def test_BiJD_all_eigenvalues():
    # Generate a symmetric matrix A
    #a = 5
    #A = np.array([[4.0, -1.0, -1.0, 0.0],
              #[-1.0, 4.0, 0.0, -1.0],
              #[-1.0, 0.0, 4.0, -1.0],
              #[0.0, -1.0, -1.0, 4.0]])
              
    # dorr
    # A = np.array([[ 6.92, -5.71,  0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.  ],
    #  [-1.21,  5.92, -4.71,  0.,    0.,    0.,    0.,    0.,    0.,    0.  ],
    #  [ 0.,   -1.21,  4.92, -3.71,  0.,    0.,    0.,    0.,    0.,    0.  ],
    #  [ 0.,    0.,   -1.21,  3.92, -2.71,  0.,    0.,    0.,    0.,    0.  ],
    #  [ 0.,    0.,    0.,   -1.21,  2.92, -1.71,  0.,    0.,    0.,    0.  ],
    #  [ 0.,    0.,    0.,    0.,    3.29, -2.08, -1.21,  0.,    0.,    0.  ],
    #  [ 0.,    0.,    0.,    0.,    0.,    2.29, -1.08, -1.21,  0.,    0.  ],
    #  [ 0.,    0.,    0.,    0.,    0.,    0.,    1.29, -0.08, -1.21,  0.  ],
    #  [ 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.29,  0.92, -1.21],
    #  [ 0.,    0.,    0.,    0.,    0.,    0.,    0.,    0.,   -0.71,  1.92]])
    
    # fiedler - symmetric
    # A = np.array([[0., 1., 2., 3., 4.],
    #  [1., 0., 1., 2., 3.],
    #  [2., 1., 0., 1., 2.],
    #  [3., 2., 1., 0., 1.],
    #  [4., 3., 2., 1., 0.]])
    
    
    # frank - ill conditioned eigenvalues
   # A = np.array([[5., 4., 3., 2., 1.], [4., 4., 3., 2., 1.], [0., 3., 3., 2., 1.], [0., 0., 2., 2., 1.], [0., 0., 0., 1., 1.]])
   
   
    # neumann - sparse
    # A = np.array([[ 4., -2., -2., -0.],
    #   [-2.,  4., -0., -2.],
    #   [-2., -0.,  4., -2.],
    #   [-0., -2., -2.,  4.]])
    
    # poisson - block tridiagonal sparse
    # A = np.array([[4.0, -1.0, -1.0, 0.0],
    #           [-1.0, 4.0, 0.0, -1.0],
    #           [-1.0, 0.0, 4.0, -1.0],
    #           [0.0, -1.0, -1.0, 4.0]])
    
    a = 10
    A = np.random.rand(a, a)
    #A = A + A.T
    # Generate an initial guess for the eigenvectors
    a = len(A[0])
    V0 = np.eye(a)[:, :a]
    W0 = np.eye(a)[:, :a]
    # Call the BiJD_all_eigenvalues function
    eigenvalues, left, right = BiJD_all_eigenvalues(A, 2, V0, W0)
    # Print the results
    print(len(eigenvalues))
    print("Eigenvalues:", eigenvalues[0])
    #print("Residuals:", left)
    true_eigenvalues = np.linalg.eigvalsh(A)
    print("true eigenvalues", true_eigenvalues)
    plot_heatmap(A)
    plot_scatter(eigenvalues[0], true_eigenvalues)
    plot_convergence(right, left)

# Run the test case
test_BiJD_all_eigenvalues()




