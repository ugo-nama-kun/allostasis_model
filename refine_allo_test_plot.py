import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from env import TEMP_SETPOINT

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})
sns.set_palette('Set1')

with open("sample_data/allostasis_test.json", mode="r") as f:
    data = json.load(f)
    
legends = ["temp", "cue", "control", "load", "memory"]

x_range = [0, 200]
# x_range = [0, len(data["temp"])]

plt.figure(figsize=(10, 5), dpi=100)

plt.subplot(3, 1, 1)
plt.plot(data["cue"][x_range[0]:x_range[1]], "r", alpha=1)
plt.plot(data["memory"][x_range[0]:x_range[1]], "blue", alpha=0.3)
plt.legend(["cue", "memory"], loc='upper right')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(3, 1, 2)
plt.plot(data["temp"][x_range[0]:x_range[1]], alpha=0.5)
plt.plot(np.arange(x_range[1]- x_range[0]), TEMP_SETPOINT * np.ones_like(data["temp"][x_range[0]:x_range[1]]), "--k", alpha=0.5)
plt.legend(["temperature"], loc='upper right')
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.subplot(3, 1, 3)
plt.plot(data["load"][x_range[0]:x_range[1]], "r", alpha=0.5)
plt.plot(data["control"][x_range[0]:x_range[1]], "blue", alpha=0.5)
plt.legend(["load", "control"], loc='upper right')
plt.ylim([-10, 10])
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("allostasis_test.pdf")
plt.show()