# test case 1
# test case
# Define your matrix A
A = np.array([[1, 2, 3], [4, 5, 6],[7,8,9]])

# Initializations
sigma = 0.5
m = 10
k = 3
V = np.eye(A.shape[0])[:, :k]  # Assuming k0 = k
W = np.eye(A.shape[0])[:, :k]  # Assuming k0 = k

# Call the BiJD function
x, y, lambda_ = BiJD(A, sigma, m, k, V, W)
print("Approximate Eigenpair (lambda, x):", lambda_, x)
