import numpy as np
import matplotlib.pyplot as plt
from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

plt.style.use("./deeplearning.mplstyle")

# ? x_train is the input variable (size in 1000 square feet)
# ? y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array(
    [
        250,
        300,
        480,
        430,
        630,
        730,
    ]
)

plt.close("all")
fig, ax, dyn_items = plt_stationary(x_train, y_train)
updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

plt_intuition(x_train, y_train)

print("x_train:", x_train)
print("y_train:", y_train)
