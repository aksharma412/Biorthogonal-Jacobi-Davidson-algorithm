import numpy as np

def BiJD(A, sigma, m, k, V, W, tolerance=1e-6):
    n = A.shape[0]
    j = len(V[0])  # Initial number of basis vectors

    # Initialize K, L, and H
    K = A @ V
    L = np.conjugate(A.T) @ W
    H = np.conjugate(W.T) @ K

    while j < m:
        # Step 1: Compute right (gi) and left (fi) eigenvectors of H
        _, g = np.linalg.eigh(H)
        f = np.linalg.inv(W) @ g
        

        # Step 2: Target the Ritz triplet (f, g, λ) with Ritz value λ closest to σ
        x = V @ g
        y = W @ f
        rr = K @ g - sigma * x
        rl = L @ f - np.conjugate(sigma) * y
        

        # Step 3: Run p steps of BCG simultaneously on the two correction equations
        # (I - xy*)(A - λI)(I - xy*) δr = rr
        # (I - yx*)(A* - λI*)(I - yx*) δl = rl
        p = 5  # Number of BCG steps, you can adjust this
        for _ in range(p):
            
            i_minus_xy_star = np.eye(n) - x @ np.conjugate(y)
            i_minus_yx_star = np.eye(n) - y @ np.conjugate(x)
            
            # Solve for delta_r using the first correction equation
            delta_r = np.linalg.solve(i_minus_xy_star @ (A - sigma * np.eye(n)) @ i_minus_xy_star, rr)
     
            # Solve for delta_l using the second correction equation
            delta_l = np.linalg.solve(i_minus_yx_star @ (np.conj(A.T) - np.conj(sigma) * np.eye(n)) @ i_minus_yx_star, rl)
    
            # Update residuals
            rr -= (A - sigma * np.eye(n)) @ (i_minus_xy_star @ delta_r)
            rl -= (np.conj(A.T) - np.conj(sigma) * np.eye(n)) @ (i_minus_yx_star @ delta_l)


        # Step 4: Update V and W
        V = np.hstack((V, delta_r))
        W = np.hstack((W, delta_l))

        # Step 5: Biorthogonalize the new basis vectors
        V, _ = np.linalg.qr(V)
        W, _ = np.linalg.qr(W)

        # Step 6: Update j, K, L, and H
        j += 1
        K = A @ V
        L = A.T @ W
        H = np.conjugate(W.T) @ K

    # Step 8: Check if convergence is achieved
    if np.linalg.norm(rr) >= tolerance:
        # Step 9: Restart
        V = x[:,:k]
        W = y[:,:k]
        K = A @ V
        L = A.T @ W
        H = np.diag(np.diag(np.conjugate(y[:,:k].T) @ K @ x[:,:k]))
        j = k
        return x, y, sigma

    return x, y, sigma


