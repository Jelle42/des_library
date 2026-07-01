import matplotlib.pyplot as plt

# Data
k = list(range(1, 33))

morning_slots = [
    1, 2, 2, 4, 4, 4, 5, 5,
    5, 6, 8, 7, 7, 7, 7, 8,
    6, 6, 7, 9, 8, 11, 10, 10,
    11, 10, 11, 12, 13, 14, 15, 16
]

value = [
    240.898, 256.061, 256.085, 250.908,
    256.566, 256.485, 256.459, 256.569,
    256.854, 256.877, 251.459, 257.149,
    257.052, 257.044, 257.038, 257.405,
    256.486, 256.475, 257.245, 251.812,
    256.853, 249.252, 249.641, 256.628,
    246.119, 256.628, 249.254, 246.307,
    238.224, 233.833, 222.988, 215.932
]

# Plot
fig, ax1 = plt.subplots(figsize=(10, 5))

# Left axis: value
ax1.plot(k, value, 'o-', linewidth=2, markersize=5, label='Value')
ax1.set_xlabel('k')
ax1.set_ylabel('Value')
ax1.set_xticks(k)
ax1.grid(True, alpha=0.3)

# # Right axis: morning slots
# ax2 = ax1.twinx()
# ax2.plot(k, morning_slots, 's--', linewidth=1.8, markersize=4, label='Morning slots')
# ax2.set_ylabel('Number of morning slots')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1, labels1, loc='best')

plt.title('Value and Number of Morning Slots vs. k')
plt.tight_layout()
plt.show()