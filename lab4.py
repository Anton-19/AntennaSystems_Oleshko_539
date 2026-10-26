import numpy as np
import matplotlib.pyplot as plt

# Вихідні дані до розрахунку (Варіант 12)
lambda_cm = 4.4
h_cm = 8.0
l_cm = 23.7

# Оптимальний коефіцієнт уповільнення (Формула 4.7)
xi = 1 + lambda_cm / (2 * l_cm)

# Кути для побудови від 0 до 90 градусів
theta = np.linspace(0, np.pi/2, 1000)
theta_deg = np.degrees(theta)

# Формула 4.3: Нормована ДС решітки з хвилею, що біжить
def F_B(th):
    num = (xi - 1) * np.sin((np.pi * l_cm / lambda_cm) * (xi - np.cos(th)))
    den = (xi - np.cos(th)) * np.sin((np.pi * l_cm / lambda_cm) * (xi - 1))
    val = np.abs(num / den)
    return np.where(np.isnan(val), 1.0, val)

Fb = F_B(theta)

# Формула 4.18/4.25: Множник грат
Fc = np.abs(np.cos((np.pi * h_cm / lambda_cm) * np.sin(theta)))

# Результ. діаграми
FH = Fb * Fc
FH = FH / np.max(FH) # Площина H

FE = Fb * np.abs(np.cos(theta))
FE = FE / np.max(FE) # Площина E

# Рисунок 1: Декартові координати
plt.figure(figsize=(10, 5))
plt.plot(theta_deg, FH, label='Площина H $F_H(\\theta)$', color='blue')
plt.plot(theta_deg, FE, label='Площина E $F_E(\\theta)$', color='orange')
plt.axhline(0.707, color='red', linestyle='--', linewidth=1, label='Рівень 0.707')

plt.title('Нормовані діаграми спрямованості')
plt.xlabel('Кут $\\theta^\\circ$')
plt.ylabel('$|F_H(\\theta)|, |F_E(\\theta)|$')
plt.xlim([0, 90])
plt.ylim([0, 1.05])
plt.xticks(np.arange(0, 95, 5))
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.show()

# Рисунок 2: Полярні координати (від -90 до 90 градусів)
theta_full = np.linspace(-np.pi/2, np.pi/2, 1000)
Fb_f = F_B(theta_full)
Fc_f = np.abs(np.cos((np.pi * h_cm / lambda_cm) * np.sin(theta_full)))

FH_f = Fb_f * Fc_f
FH_f = FH_f / np.max(FH_f)

FE_f = Fb_f * np.abs(np.cos(theta_full))
FE_f = FE_f / np.max(FE_f)

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)
ax.plot(theta_full, FH_f, label='Площина H', color='blue')
ax.plot(theta_full, FE_f, label='Площина E', color='orange')

ax.set_theta_zero_location("N")
ax.set_thetamin(-90)
ax.set_thetamax(90)
plt.title('Радіальна діаграма спрямованості FH та FE', pad=20)
plt.legend(loc='lower left')
plt.show()