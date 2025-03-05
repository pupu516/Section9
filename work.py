import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt

def construct_heisenberg_xxx_hamiltonian(N, J=1.0):
    """Constructs the Heisenberg XXX Hamiltonian matrix for N spins with periodic boundary conditions."""
    dim = 2**N  # Hilbert space dimension
    H = sp.lil_matrix((dim, dim), dtype=np.float64)
    
    for i in range(N):
        j = (i + 1) % N  # Periodic boundary condition
        
        for state in range(dim):
            spin_i = (state >> i) & 1  # Extract spin at site i
            spin_j = (state >> j) & 1  # Extract spin at site j
            
            # Sz_i * Sz_j term
            H[state, state] += J * (0.25 if spin_i == spin_j else -0.25)
            
            # S+_i S-_j and S-_i S+_j terms (flipping spins)
            if spin_i != spin_j:
                flipped_state = state ^ (1 << i) ^ (1 << j)  # Flip bits at i and j
                H[state, flipped_state] += J * 0.5
    
    return H.tocsr()  # Convert to CSR format for efficiency

def qr_algorithm(H, tol=1e-9, max_iter=1000):
    """Computes the eigenvalues of the Heisenberg XXX Hamiltonian using the QR algorithm."""
    H = H.toarray()  # Convert sparse matrix to dense for QR iteration
    n = H.shape[0]
    Q_total = np.eye(n)  # Store cumulative Q matrix
    
    for _ in range(max_iter):
        Q, R = np.linalg.qr(H)  # Perform QR decomposition
        H = R @ Q  # Update matrix
        Q_total = Q_total @ Q  # Accumulate Q matrices
        
        # Check for convergence (off-diagonal elements are small)
        off_diag_norm = np.linalg.norm(H - np.diag(np.diagonal(H)))
        if off_diag_norm < tol:
            break
    
    eigenvalues = np.diagonal(H)  # Extract eigenvalues
    return np.sort(eigenvalues)

def plot_eigenvalues(N):
    """Computes and plots the eigenvalues of the Heisenberg XXX Hamiltonian."""
    H = construct_heisenberg_xxx_hamiltonian(N)
    eigenvalues = qr_algorithm(H)
    
    plt.figure(figsize=(8,6))
    plt.plot(range(len(eigenvalues)), eigenvalues, 'bo-', label='Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.title(f'Heisenberg XXX Hamiltonian Eigenvalues (N={N})')
    plt.legend()
    plt.grid()
    plt.savefig(f'eigenvalues_N{N}.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    N = 3  # Number of spins
    plot_eigenvalues(N)

