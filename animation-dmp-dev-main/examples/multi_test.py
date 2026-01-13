import numpy as np
import matplotlib.pyplot as plt

# ---- Simulated DMPs ----

# Time setup
T = 1.0
dt = 0.01
timesteps = int(T / dt)
time = np.linspace(0, T, timesteps)

# Simulated phase variable s (assume canonical system with decay)
alpha_s = 4.0
s = np.exp(-alpha_s * time)

# Simulate forcing terms for 3 DMPs
# DMP A: characteristic curved motion
f1 = np.sin(2 * np.pi * time) * 2.0

# DMP B: linear motion (approximated forcing term)
f2 = np.ones_like(time) * 0.5

# DMP C: zero DMP
f3 = np.zeros_like(time)

# Stack them for easier processing
forcing_terms = np.stack([f1, f2, f3])

# ---- Automatic Weighting via Energy ----

# Compute energy of each DMP
energies = np.sum(forcing_terms**2, axis=1)
weights = energies / np.sum(energies)

print("Energies:", energies)
print("Auto-computed weights:", weights)

# ---- Goal values for each DMP ----
goals = np.array([2.0, 3.0, 0.0])
g_blend = np.sum(weights * goals)
print("Blended goal:", g_blend)

# ---- Blend forcing terms ----
f_blend = np.sum(weights[:, None] * forcing_terms, axis=0)

# ---- Integrate DMP dynamics ----
# Basic second-order system parameters
K = 1000
D = 2 * np.sqrt(K)
tau = 1.0

y = 0.0
dy = 0.0
ddy = 0.0

y_track = []

for t in range(timesteps):
    forcing = f_blend[t]
    accel = (K * (g_blend - y) - D * dy + forcing) / tau
    dy += accel * dt
    y += dy * dt
    y_track.append(y)

y_track = np.array(y_track)

# ---- Plot ----
plt.figure(figsize=(10, 5))
plt.plot(time, f1, label="DMP A (curve)")
plt.plot(time, f2, label="DMP B (linear)")
plt.plot(time, f3, label="DMP C (zero)")
plt.plot(time, f_blend, label="Blended Forcing", linewidth=2)
plt.legend()
plt.title("Forcing Terms")

plt.figure(figsize=(10, 5))
plt.plot(time, y_track, label="Blended Trajectory", color='black', linewidth=2)
plt.axhline(g_blend, linestyle='--', color='gray', label="Blended Goal")
plt.legend()
plt.title("Blended DMP Output")
plt.xlabel("Time")
plt.ylabel("y")
plt.grid(True)
plt.show()
