import numpy as np
import matplotlib.pyplot as plt

# --- Вихідні дані (Варіант 12) ---
variant = 12
lambda_cm = 3.6  # см
a_p = 15.0  # см
b_p = 15.0  # см

# Масив кутів θ від 0.0001° до 90°
steps = np.arange(0.0001, 90.0, 0.001)
theta_rad = np.radians(steps)

# --- Обчислення нормованих ДС ---
# Площина H
u_H = (np.pi * a_p / lambda_cm) * np.sin(theta_rad)
denom_H = 1.0 - (2.0 * a_p / lambda_cm * np.sin(theta_rad)) ** 2

with np.errstate(divide='ignore', invalid='ignore'):
    F_H_ap = np.where(np.abs(denom_H) < 1e-6, np.pi / 4.0, np.cos(u_H) / denom_H)

F_H = np.abs(F_H_ap) * (1.0 + np.cos(theta_rad)) / 2.0
F_H = F_H / np.max(F_H)

# Площина E
u_E = (np.pi * b_p / lambda_cm) * np.sin(theta_rad)

with np.errstate(divide='ignore', invalid='ignore'):
    F_E_ap = np.where(np.abs(u_E) < 1e-6, 1.0, np.sin(u_E) / u_E)

F_E = np.abs(F_E_ap) * (1.0 + np.cos(theta_rad)) / 2.0
F_E = F_E / np.max(F_E)

# --- Пошук рівня половинної потужності (0.707) ---
idx_H_0707 = np.argmin(np.abs(F_H - 0.7071))
theta_H_0707 = steps[idx_H_0707]
val_H_0707 = F_H[idx_H_0707]

idx_E_0707 = np.argmin(np.abs(F_E - 0.7071))
theta_E_0707 = steps[idx_E_0707]
val_E_0707 = F_E[idx_E_0707]

print(f"Ширина ДС в площині H (2*θ_05) = {2 * theta_H_0707:.2f}°")
print(f"Ширина ДС в площині E (2*θ_05) = {2 * theta_E_0707:.2f}°")

# --- Пошук локальних екстремумів ---
max_x_FH, max_y_FH, min_x_FH, min_y_FH = [], [], [], []
max_x_FE, max_y_FE, min_x_FE, min_y_FE = [], [], [], []

for i in range(1, len(steps) - 1):
    if F_H[i] > F_H[i - 1] and F_H[i] > F_H[i + 1]:
        max_x_FH.append(steps[i])
        max_y_FH.append(F_H[i])
    if F_H[i] < F_H[i - 1] and F_H[i] < F_H[i + 1]:
        min_x_FH.append(steps[i])
        min_y_FH.append(F_H[i])

    if F_E[i] > F_E[i - 1] and F_E[i] > F_E[i + 1]:
        max_x_FE.append(steps[i])
        max_y_FE.append(F_E[i])
    if F_E[i] < F_E[i - 1] and F_E[i] < F_E[i + 1]:
        min_x_FE.append(steps[i])
        min_y_FE.append(F_E[i])

# --- Побудова прямокутного графіка ---
fig, ax = plt.subplots(figsize=(20 / 2.54, 12 / 2.54))

ax.plot(steps, F_H, linewidth=1.2, color='blue', label="$F_H(\\theta)$")
ax.plot(steps, F_E, linewidth=1.2, color='green', label="$F_E(\\theta)$")

ax.plot(theta_H_0707, val_H_0707, 'ro', markersize=4, label="Рівень 0.707 H")
ax.plot(theta_E_0707, val_E_0707, 'go', markersize=4, label="Рівень 0.707 E")

plt.annotate(f'({val_H_0707:.3f}, {theta_H_0707:.2f}°)', xy=(theta_H_0707, val_H_0707),
             xytext=(theta_H_0707 + 3, val_H_0707 + 0.05),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=7)

plt.annotate(f'({val_E_0707:.3f}, {theta_E_0707:.2f}°)', xy=(theta_E_0707, val_E_0707),
             xytext=(theta_E_0707 - 12, val_E_0707 + 0.05),
             arrowprops=dict(arrowstyle='->', color='black'), fontsize=7)

ax.plot([0, theta_H_0707], [val_H_0707, val_H_0707], 'r--', linewidth=0.5)
ax.plot([theta_H_0707, theta_H_0707], [0, val_H_0707], 'r--', linewidth=0.5)
ax.plot([theta_E_0707, theta_E_0707], [0, val_E_0707], 'g--', linewidth=0.5)

ax.plot(max_x_FH, max_y_FH, "o", markersize=4, color="black", label="$\\theta_{max}$ FH")
ax.plot(max_x_FE, max_y_FE, "o", markersize=4, color="grey", label="$\\theta_{max}$ FE")
ax.plot(min_x_FH, min_y_FH, "o", markersize=4, color="blue", label="$\\theta_{min}$ FH")
ax.plot(min_x_FE, min_y_FE, "o", markersize=4, color="magenta", label="$\\theta_{min}$ FE")

ax.set_xlabel('θ, °', fontsize=10)
ax.set_ylabel('|F_H(θ)|, |F_E(θ)|', fontsize=10)
plt.xticks(np.arange(0, 95, 5), fontsize=7)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=7)
plt.ylim(-0.01, 1.05)
plt.xlim(0, 90.5)
plt.legend(loc="upper right", fontsize=7)
plt.grid(which='both', linestyle='--', linewidth=0.3, color='gray')

fig.savefig("ДС_рупор_прямокутна,пр3.png", dpi=600)
plt.show()

# --- Побудова радіального (полярного) графіка ---
fig2 = plt.figure(figsize=(20 / 2.54, 12 / 2.54))
ax_polar = fig2.add_subplot(111, projection='polar')

theta_rad_full = np.radians(steps)
ax_polar.plot(theta_rad_full, F_H, linewidth=1.0, color="blue", label="$F_H(\\theta)$")
ax_polar.plot(theta_rad_full, F_E, linewidth=1.0, color="green", label="$F_E(\\theta)$")
ax_polar.plot(-theta_rad_full, F_H, linewidth=1.0, color="blue", alpha=0.7)
ax_polar.plot(-theta_rad_full, F_E, linewidth=1.0, color="green", alpha=0.7)

ax_polar.set_theta_zero_location("N")
ax_polar.set_theta_direction(-1)
ax_polar.set_thetagrids(range(0, 181, 10))
ax_polar.set_title("Радіальна діаграма спрямованості рупорної антени $F_H$ та $F_E$", fontsize=11)
ax_polar.set_rlabel_position(90)
ax_polar.grid(True, linestyle='--', linewidth=0.3)

for label in ax_polar.get_xticklabels():
    label.set_fontsize(7)

ax_polar.legend(loc="lower left", fontsize=8)
fig2.savefig("ДС_рупор_радіальна,пр3.png", dpi=600)
plt.show()