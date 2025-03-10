import numpy as np
import scipy.integrate as spi

# Define constants
k_B = 1.38064852e-23  # Boltzmann constant (J/K)
h = 6.626e-34  # Planck constant (J·s)
c = 3.0e8  # Speed of light (m/s)
pi = np.pi

# Prefactor for Stefan-Boltzmann constant
prefactor = (2 * pi * k_B**4) / (c**2 * (h / (2 * pi))**3)

def black_body_integral():
    """
    Evaluates the integral for black body radiation using the variable transformation:
    x = z / (1 - z)
    """
    def integrand(z):
        x = z / (1 - z)  # Change of variable x = z / (1 - z)
        dx_dz = 1 / (1 - z)**2  # Derivative dx/dz
        return (x**3 / (np.exp(x) - 1)) * dx_dz  # Apply transformation

    # Perform numerical integration from 0 to 1
    integral_result, error = spi.quad(integrand, 0, 1)
    return integral_result

# Compute the integral value
integral_value = black_body_integral()

# Multiply by the prefactor to get the final W value
W_value = prefactor * integral_value

# Display the result
print(f"Calculated W value: {W_value}")

