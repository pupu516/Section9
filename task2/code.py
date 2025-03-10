###---------------------------------
### Part a
import numpy as np
import matplotlib.pyplot as plt

# Initial conditions
e = 0.6
q1_0 = 1 - e
q2_0 = 0
p1_0 = 0
p2_0 = np.sqrt((1 + e) / (1 - e))

# Time parameters
Tf = 200
N = 100000
dt = Tf / N

# Arrays to store positions
q1 = np.zeros(N)
q2 = np.zeros(N)
p1 = np.zeros(N)
p2 = np.zeros(N)

# Initial conditions
q1[0] = q1_0
q2[0] = q2_0
p1[0] = p1_0
p2[0] = p2_0

# Explicit Euler method
for i in range(N - 1):
    r = np.sqrt(q1[i]**2 + q2[i]**2)
    q1[i + 1] = q1[i] + dt * p1[i]
    q2[i + 1] = q2[i] + dt * p2[i]
    p1[i + 1] = p1[i] - dt * q1[i] / r**3
    p2[i + 1] = p2[i] - dt * q2[i] / r**3

# Plot the orbit
plt.figure(figsize=(8, 8))
plt.plot(q1, q2, label="Explicit Euler")
plt.xlabel("q1")
plt.ylabel("q2")
plt.title("Planet Orbit (Explicit Euler)")
plt.legend()
plt.savefig("explicit_euler_orbit.png")



###---------------------------------
### Part b
# Symplectic Euler method
q1_sym = np.zeros(N)
q2_sym = np.zeros(N)
p1_sym = np.zeros(N)
p2_sym = np.zeros(N)

# Initial conditions
q1_sym[0] = q1_0
q2_sym[0] = q2_0
p1_sym[0] = p1_0
p2_sym[0] = p2_0

for i in range(N - 1):
    r = np.sqrt(q1_sym[i]**2 + q2_sym[i]**2)
    p1_sym[i + 1] = p1_sym[i] - dt * q1_sym[i] / r**3
    p2_sym[i + 1] = p2_sym[i] - dt * q2_sym[i] / r**3
    q1_sym[i + 1] = q1_sym[i] + dt * p1_sym[i + 1]
    q2_sym[i + 1] = q2_sym[i] + dt * p2_sym[i + 1]

# Plot both orbits
plt.figure(figsize=(8, 8))
plt.plot(q1, q2, label="Explicit Euler")
plt.plot(q1_sym, q2_sym, label="Symplectic Euler")
plt.xlabel("q1")
plt.ylabel("q2")
plt.title("Planet Orbit Comparison")
plt.legend()
plt.savefig("orbit_comparison.png")


