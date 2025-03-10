###-------------------------------
### Part a
import numpy as np
import matplotlib.pyplot as plt

# Cost function
def H(theta):
    return theta**4 - 8 * theta**2 - 2 * np.cos(4 * np.pi * theta)

# Gradient of the cost function
def grad_H(theta):
    return 4 * theta**3 - 16 * theta + 8 * np.pi * np.sin(4 * np.pi * theta)

# Gradient descent
def gradient_descent(theta0, learning_rate, max_steps):
    theta = theta0
    history = []
    for _ in range(max_steps):
        theta = theta - learning_rate * grad_H(theta)
        history.append(theta)
    return np.array(history)

# Initial guesses
theta0_list = [-1, 0.5, 3]
learning_rate = 0.01
max_steps = 1000

# Plot the cost function
theta_values = np.linspace(-3, 3, 1000)
plt.plot(theta_values, H(theta_values), label="H(θ)")
for theta0 in theta0_list:
    history = gradient_descent(theta0, learning_rate, max_steps)
    plt.plot(history, H(history), 'ro', markersize=2, label=f"θ0 = {theta0}")
plt.xlabel("θ")
plt.ylabel("H(θ)")
plt.title("Gradient Descent")
plt.legend()
plt.savefig("gradient_descent.png")




###-------------------------------
### Part b

def metropolis_hastings(theta0, beta, sigma, max_steps):
    theta = theta0
    history = []
    for _ in range(max_steps):
        theta_star = theta + np.random.normal(0, sigma)
        delta_H = H(theta_star) - H(theta)
        r = np.exp(-beta * delta_H)
        if r > 1 or np.random.rand() < r:
            theta = theta_star
        history.append(theta)
    return np.array(history)

# Parameters
beta = 1.0
sigma = 0.1
max_steps = 10000

# Run Metropolis-Hastings
for theta0 in theta0_list:
    history = metropolis_hastings(theta0, beta, sigma, max_steps)
    plt.plot(history, H(history), 'bo', markersize=2, label=f"θ0 = {theta0}")
plt.xlabel("θ")
plt.ylabel("H(θ)")
plt.title("Metropolis-Hastings")
plt.legend()
plt.savefig("metropolis_hastings.png")


###-------------------------------
### Part c
def simulated_annealing(theta0, beta0, delta, sigma, max_steps):
    theta = theta0
    beta = beta0
    history = []
    for i in range(max_steps):
        theta_star = theta + np.random.normal(0, sigma)
        delta_H = H(theta_star) - H(theta)
        r = np.exp(-beta * delta_H)
        if r > 1 or np.random.rand() < r:
            theta = theta_star
        history.append(theta)
        beta += delta
    return np.array(history)

# Parameters
beta0 = 1.0
delta = 0.001
sigma = 0.1
max_steps = 10000

# Run Simulated Annealing
for theta0 in theta0_list:
    history = simulated_annealing(theta0, beta0, delta, sigma, max_steps)
    plt.plot(history, H(history), 'go', markersize=2, label=f"θ0 = {theta0}")
plt.xlabel("θ")
plt.ylabel("H(θ)")
plt.title("Simulated Annealing")
plt.legend()
plt.savefig("simulated_annealing.png")
