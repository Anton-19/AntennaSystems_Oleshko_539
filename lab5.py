import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import brentq

# Початкові дані (Варіант 12, Вид 3)
lam = 3.1
a = 2.3
N = 6
d = 3.2
nu = 0

# Параметри хвилеводу
lam_kr = 2 * a
lam_xv = lam / np.sqrt(1 - (lam / lam_kr)**2)
k = 2 * np.pi / lam
xi = (2 * np.pi * d / lam_xv) - nu * np.pi

def calc_Fc(theta_rad):
    psi = k * d * np.sin(theta_rad) - xi
    num = np.sin(N * psi / 2)
    den = N * np.sin(psi / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(den) < 1e-12, 1.0, np.abs(num / den))

def calc_FE(theta_rad):
    return calc_Fc(theta_rad)

def calc_FH(theta_rad):
    cos_th = np.cos(theta_rad)
    with np.errstate(divide='ignore', invalid='ignore'):
        F1_H = np.where(
            np.abs(cos_th) < 1e-12,
            0.0,
            np.abs(np.cos((np.pi / 2) * np.sin(theta_rad)) / cos_th)
        )
    return F1_H * calc_Fc(theta_rad)

#  Розрахунок ДС у межах [-90°, +90°]
theta_deg = np.linspace(-90, 90, 18001)
theta_rad = np.radians(theta_deg)
FE = calc_FE(theta_rad)
FH = calc_FH(theta_rad)

# Межі головної пелюстки за рівнем 0.707
tH_l = np.degrees(brentq(lambda th: calc_FH(th) - 0.707, np.radians(-20), np.radians(-13.3)))
tH_r = np.degrees(brentq(lambda th: calc_FH(th) - 0.707, np.radians(-13.3), np.radians(-5)))
tE_l = np.degrees(brentq(lambda th: calc_FE(th) - 0.707, np.radians(-20), np.radians(-13.3)))
tE_r = np.degrees(brentq(lambda th: calc_FE(th) - 0.707, np.radians(-13.3), np.radians(-5)))

print(f"Рівень 0.707 H: ({tH_l:.2f}°, {tH_r:.2f}°), ширина = {tH_r - tH_l:.2f}°")
print(f"Рівень 0.707 E: ({tE_l:.2f}°, {tE_r:.2f}°), ширина = {tE_r - tE_l:.2f}°")

# Максимуми та нулі ДС
max_H, _ = find_peaks(FH, prominence=0.01)
min_H, _ = find_peaks(-FH)
min_H = np.unique(np.concatenate(([0], min_H[FH[min_H] < 1e-3], [len(theta_deg) - 1])))

max_E, _ = find_peaks(FE, prominence=0.01)
min_E, _ = find_peaks(-FE)
min_E = min_E[FE[min_E] < 1e-3]

# Декартова система координат
plt.figure(figsize=(14, 6))
plt.plot(theta_deg, FH, label='$F_H(\\theta)$', color='tab:blue', linewidth=1.8)
plt.plot(theta_deg, FE, label='$F_E(\\theta)$', color='tab:orange', linewidth=1.8)

plt.plot(theta_deg[max_H], FH[max_H], 's', color='firebrick', markersize=5, label='Максимуми H')
plt.plot(theta_deg[min_H], FH[min_H], 'o', color='black', markersize=5, label='Мінімуми H')
plt.plot(theta_deg[max_E], FE[max_E], '^', color='tab:orange', markersize=6, label='Максимуми E')
plt.plot(theta_deg[min_E], FE[min_E], 'x', color='tab:orange', markersize=6, label='Мінімуми E')

plt.plot([tH_l, tH_r], [0.707, 0.707], 'o', markerfacecolor='none', markeredgecolor='tab:blue',
         markersize=8, label=f'Рівень 0.707 H ({tH_l:.2f}°, {tH_r:.2f}°)')
plt.plot([tE_l, tE_r], [0.707, 0.707], 'o', color='tab:orange',
         markersize=5, label=f'Рівень 0.707 E ({tE_l:.2f}°, {tE_r:.2f}°)')
plt.plot([tE_l, tE_r], [0.707, 0.707], 'k-', linewidth=0.8)

plt.title('Діаграма спрямованості хвилевідно-щілинної антени (Варіант 12, Вид 3)', fontsize=12)
plt.xlabel('Кут $\\theta$, °')
plt.ylabel('Нормована амплітуда $|F(\\theta)|$')
plt.xlim([-90, 90])
plt.ylim([-0.02, 1.05])
plt.xticks(np.arange(-90, 91, 10))
plt.yticks(np.arange(0.0, 1.05, 0.1))
plt.grid(True, linestyle='-', alpha=0.35)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=9)
plt.tight_layout()
plt.savefig('cartesian_patterns_lab5.png', dpi=300, bbox_inches='tight')
plt.show()

# Полярна система координат
plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.plot(theta_rad, FH, label='$F_H(\\theta)$', color='tab:blue', linewidth=2)
ax.plot(theta_rad, FE, label='$F_E(\\theta)$', color='tab:orange', linewidth=2)
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_thetamin(-90)
ax.set_thetamax(90)
plt.title('Радіальна ДС ХвЩА в площинах E та H', pad=20, fontsize=12)
plt.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig('polar_patterns_lab5.png', dpi=300)
plt.show()