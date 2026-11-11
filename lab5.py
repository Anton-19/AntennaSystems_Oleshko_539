import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Початкові дані (Варіант 12, Вид 3) ---
lambda_cm = 3.1  # Довжина хвилі [см]
N = 6  # Кількість щілин
l_cm = 1.6  # Довжина щілини [см]
d_cm = 3.2  # Відстань між щілинами [см]

k = 2 * np.pi / lambda_cm

# Сітка кутів від -90 до +90 градусів
theta_deg = np.linspace(-90, 90, 3000)
theta_rad = np.radians(theta_deg)

# --- Розрахунок ДС ---
# В E-площині (уздовж осі хвилеводу, множник решітки)
psi = k * d_cm * np.sin(theta_rad)
with np.errstate(divide='ignore', invalid='ignore'):
    Fc = np.abs(np.sin(N * psi / 2) / (N * np.sin(psi / 2)))
    Fc[np.abs(psi / 2) < 1e-7] = 1.0
    Fc[np.isnan(Fc)] = 1.0

FE = np.cos(theta_rad) * Fc
FE /= np.max(FE)

# H-площині
FH = np.abs(np.cos((np.pi * l_cm / lambda_cm) * np.sin(theta_rad)))
FH /= np.max(FH)


# --- Функція пошуку характеристик ---
def analyze_pattern(F, th_deg):
    idx_0707 = np.where((F <= 0.707) & (th_deg >= 0))[0]
    th_0707 = th_deg[idx_0707[0]] if len(idx_0707) > 0 else 0

    # Нулі ДС (мінімуми)
    min_idx, _ = find_peaks(-F)
    zeros = [i for i in min_idx if F[i] < 0.05 and th_deg[i] >= 0]

    # Максимуми бічних пелюсток
    max_idx, _ = find_peaks(F)
    lobes = [i for i in max_idx if th_deg[i] > th_0707]

    return th_0707, zeros, lobes


th_E, zE, lE = analyze_pattern(FE, theta_deg)
th_H, zH, lH = analyze_pattern(FH, theta_deg)

print(f"E-plane width (2*theta_0707): {2 * th_E:.2f} deg")
print(f"H-plane width (2*theta_0707): {2 * th_H:.2f} deg")

# --- Побудова у декартових координатах ---
plt.figure(figsize=(11, 6))
plt.plot(theta_deg, FE, label='Площина E $F_E(\\theta)$', color='darkorange', linewidth=2)
plt.plot(theta_deg, FH, label='Площина H $F_H(\\theta)$', color='blue', linewidth=2)
plt.axhline(0.707, color='red', linestyle='--', label='Рівень 0.707')

# Позначки нулів та максимумів БП для E-площини
plt.scatter(theta_deg[zE], FE[zE], color='green', marker='x', s=70, zorder=5, label='Нулі $F_E(\\theta)$')
plt.scatter(theta_deg[lE], FE[lE], color='red', marker='^', s=70, zorder=5, label='Макс. БП $F_E(\\theta)$')

for i in lE:
    plt.annotate(f"({theta_deg[i]:.1f}°, {FE[i]:.2f})", (theta_deg[i], FE[i]),
                 xytext=(0, 8), textcoords="offset points", ha='center', fontsize=8, color='red')

for i in zE:
    plt.annotate(f"{theta_deg[i]:.1f}°", (theta_deg[i], FE[i]),
                 xytext=(0, -14), textcoords="offset points", ha='center', fontsize=7, color='green')

plt.title('Нормовані діаграми спрямованості ХвЩА (Варіант 12, Вид 3)', fontsize=12, fontweight='bold')
plt.xlabel('Кут $\\theta^\\circ$');
plt.ylabel('$|F(\\theta)|$')
plt.xlim([-90, 90]);
plt.ylim([0, 1.05])
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('cartesian_patterns_lab5.png', dpi=300)
plt.show()

# --- Побудова у полярній системі координат ---
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.plot(theta_rad, FE, label='Площина E $F_E(\\theta)$', color='darkorange', linewidth=2)
ax.plot(theta_rad, FH, label='Площина H $F_H(\\theta)$', color='blue', linewidth=2)
ax.set_theta_zero_location("N")
ax.set_thetamin(-90)
ax.set_thetamax(90)
plt.title('Радіальна ДС ХвЩА в плочинах E та H', pad=20, fontsize=12, fontweight='bold')
plt.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig('polar_patterns_lab5.png', dpi=300)
plt.show()