import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Початкові дані (Варіант 12) ---
lambda_cm = 4.4
h_cm = 8.0
l_cm = 23.7

# Оптимальний коефіцієнт уповільнення
xi = 1 + lambda_cm / (2 * l_cm)

# Сітка кутів (від 0 до 90 градусів)
theta_deg = np.linspace(0, 90, 2000)
theta_rad = np.radians(theta_deg)


# Множник хвилі, що біжить F_B(theta) Формула 4.3)
def F_B(th):
    num = (xi - 1) * np.sin((np.pi * l_cm / lambda_cm) * (xi - np.cos(th)))
    den = (xi - np.cos(th)) * np.sin((np.pi * l_cm / lambda_cm) * (xi - 1))
    val = np.abs(num / den)
    return np.where(np.isnan(val) | np.isinf(val), 1.0, val)


Fb = F_B(theta_rad)

#  Однострижнева антена
F1_E = Fb * np.abs(np.cos(theta_rad))
F1_E /= np.max(F1_E)

F1_H = Fb.copy()
F1_H /= np.max(F1_H)

#  Двострижнева антена
Fc = np.abs(np.cos((np.pi * h_cm / lambda_cm) * np.sin(theta_rad)))  # Множник решітки
F2_E = F1_E.copy()  # У площині E решітка не змінює ДС
F2_H = F1_H * Fc
F2_H /= np.max(F2_H)


# Функція для пошуку точок нулів та максимумів БП
def analyze_pattern(F):
    idx_0707 = np.where(F <= 0.707)[0]
    th_0707 = theta_deg[idx_0707[0]] if len(idx_0707) > 0 else 0

    # Нулі (місцеві мінімуми)
    min_idx, _ = find_peaks(-F)
    zeros = [i for i in min_idx if F[i] < 0.05]

    # Бічні пелюстки (місцеві максимуми поза головним променем)
    max_idx, _ = find_peaks(F)
    lobes = [i for i in max_idx if theta_deg[i] > th_0707]

    return zeros, lobes


# --- Побудова графіків у декартових координатах ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

#  Однострижнева ДСА
z1E, l1E = analyze_pattern(F1_E)
z1H, l1H = analyze_pattern(F1_H)

ax1.plot(theta_deg, F1_E, label='Площина E $F_{1E}(\\theta)$', color='darkorange', linewidth=2)
ax1.plot(theta_deg, F1_H, label='Площина H $F_{1H}(\\theta)$', color='blue', linewidth=2)
ax1.axhline(0.707, color='red', linestyle='--', label='Рівень 0.707')

# Позначення нулів та максимумів БП
ax1.scatter(theta_deg[z1E], F1_E[z1E], color='green', marker='x', s=70, zorder=5, label='Нулі ДС')
ax1.scatter(theta_deg[l1E], F1_E[l1E], color='red', marker='^', s=70, zorder=5, label='Макс. БП (E)')
ax1.scatter(theta_deg[l1H], F1_H[l1H], color='purple', marker='s', s=50, zorder=5, label='Макс. БП (H)')

# Аннотація ключових точок
for i in l1E[:2]:
    ax1.annotate(f"({theta_deg[i]:.1f}°, {F1_E[i]:.2f})", (theta_deg[i], F1_E[i]),
                 xytext=(0, 8), textcoords="offset points", ha='center', fontsize=8, color='red')
for i in z1E[:2]:
    ax1.annotate(f"{theta_deg[i]:.1f}°", (theta_deg[i], F1_E[i]),
                 xytext=(0, -14), textcoords="offset points", ha='center', fontsize=8, color='green')

ax1.set_title('Однострижнева ДСА: ДС у площинах E та H', fontsize=12, fontweight='bold')
ax1.set_xlabel('Кут $\\theta^\\circ$');
ax1.set_ylabel('$|F(\\theta)|$')
ax1.set_xlim([0, 90]);
ax1.set_ylim([0, 1.05]);
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper right')

#  Двострижнева ДСА
z2H, l2H = analyze_pattern(F2_H)

ax2.plot(theta_deg, F2_E, label='Площина E $F_{2E}(\\theta)$', color='darkorange', linestyle='--', linewidth=1.8)
ax2.plot(theta_deg, F2_H, label='Площина H $F_{2H}(\\theta)$', color='blue', linewidth=2)
ax2.axhline(0.707, color='red', linestyle='--', label='Рівень 0.707')

ax2.scatter(theta_deg[z2H], F2_H[z2H], color='green', marker='x', s=70, zorder=5, label='Нулі ДС (H)')
ax2.scatter(theta_deg[l2H], F2_H[l2H], color='red', marker='^', s=70, zorder=5, label='Макс. БП (H)')

for i in l2H[:3]:
    ax2.annotate(f"({theta_deg[i]:.1f}°, {F2_H[i]:.2f})", (theta_deg[i], F2_H[i]),
                 xytext=(0, 8), textcoords="offset points", ha='center', fontsize=8, color='red')
for i in z2H[:3]:
    ax2.annotate(f"{theta_deg[i]:.1f}°", (theta_deg[i], F2_H[i]),
                 xytext=(0, -14), textcoords="offset points", ha='center', fontsize=8, color='green')

ax2.set_title('Двострижнева ДСА ($h = 8$ см): ДС у площинах E та H', fontsize=12, fontweight='bold')
ax2.set_xlabel('Кут $\\theta^\\circ$');
ax2.set_ylabel('$|F(\\theta)|$')
ax2.set_xlim([0, 90]);
ax2.set_ylim([0, 1.05]);
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('cartesian_patterns.png', dpi=300)
plt.show()

# --- Побудова у полярних координатах (-90° до +90°) ---
theta_full = np.linspace(-np.pi / 2, np.pi / 2, 2000)
Fb_f = F_B(theta_full)
F1_E_f = Fb_f * np.abs(np.cos(theta_full))
F1_E_f /= np.max(F1_E_f)
F1_H_f = Fb_f / np.max(Fb_f)

Fc_f = np.abs(np.cos((np.pi * h_cm / lambda_cm) * np.sin(theta_full)))
F2_H_f = F1_H_f * Fc_f
F2_H_f /= np.max(F2_H_f)

plt.figure(figsize=(10, 5))

ax_p1 = plt.subplot(121, polar=True)
ax_p1.plot(theta_full, F1_E_f, label='E-площина', color='darkorange')
ax_p1.plot(theta_full, F1_H_f, label='H-площина', color='blue')
ax_p1.set_theta_zero_location("N");
ax_p1.set_thetamin(-90);
ax_p1.set_thetamax(90)
ax_p1.set_title('Однострижнева ДСА', pad=15)
ax_p1.legend(loc='lower left', fontsize=8)

ax_p2 = plt.subplot(122, polar=True)
ax_p2.plot(theta_full, F1_E_f, label='E-площина', color='darkorange', linestyle='--')
ax_p2.plot(theta_full, F2_H_f, label='H-площина', color='blue')
ax_p2.set_theta_zero_location("N");
ax_p2.set_thetamin(-90);
ax_p2.set_thetamax(90)
ax_p2.set_title('Двострижнева ДСА ($h=8$ см)', pad=15)
ax_p2.legend(loc='lower left', fontsize=8)

plt.tight_layout()
plt.savefig('polar_patterns.png', dpi=300)
plt.show()