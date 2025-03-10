### Part a
###--------------------------------------------------
import numpy as np
from scipy.integrate import quad

# Constants
k = 1.38064852e-23  # Boltzmann constant, J/K
h = 6.626e-34       # Planck's constant, J·s
c = 3e8             # Speed of light, m/s
hbar = h / (2 * np.pi)

# Prefactor
prefactor = (k**4) / (c**2 * hbar**3 * 4 * np.pi**2)

# Integrand function
def integrand(t):
    x = t / (1 - t)
    return (x**3 / (np.exp(x) - 1)) * (1 / (1 - t)**2)

# Evaluate the integral
integral_value, _ = quad(integrand, 0, 1)
sigma = 2 * np.pi * prefactor * integral_value

print(f"Calculated Stefan-Boltzmann constant: {sigma:.6e} W/m^2K^4")

###--------------------------------------------------
### Part b

from scipy.integrate import fixed_quad

# Evaluate the integral using fixed_quad
integral_value_fixed, _ = fixed_quad(integrand, 0, 1, n=100)
sigma_fixed = 2 * np.pi * prefactor * integral_value_fixed

print(f"Stefan-Boltzmann constant (fixed_quad): {sigma_fixed:.6e} W/m^2K^4")

###--------------------------------------------------
### Part c

# Integrand function for quad
def integrand_quad(x):
    return x**3 / (np.exp(x) - 1)

# Evaluate the integral using quad
integral_value_quad, _ = quad(integrand_quad, 0, np.inf)
sigma_quad = 2 * np.pi * prefactor * integral_value_quad

print(f"Stefan-Boltzmann constant (quad): {sigma_quad:.6e} W/m^2K^4")





