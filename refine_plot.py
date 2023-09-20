import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from env import TEMP_SETPOINT

sns.set()
sns.set_context("paper")

with open("sample_data/allostasis.json", mode="r") as f:
    data = json.load(f)
    
legends = ["temp", "cue", "control", "load", "memory"]

x_range = [250, 425]
# x_range = [0, len(data["temp"])]

plt.figure(figsize=(8, 4), dpi=100)

plt.subplot(3, 1, 1)
plt.plot(data["temp"][x_range[0]:x_range[1]], alpha=0.5)
plt.plot(np.arange(x_range[1]- x_range[0]), TEMP_SETPOINT * np.ones_like(data["temp"][x_range[0]:x_range[1]]), "--k", alpha=0.5)
plt.legend(["temperature"])

plt.subplot(3, 1, 2)
plt.plot(data["cue"][x_range[0]:x_range[1]], "r", alpha=1)
plt.plot(data["memory"][x_range[0]:x_range[1]], "blue", alpha=0.3)
plt.legend(["cue-presentation", "memory"])

plt.subplot(3, 1, 3)
plt.plot(data["load"][x_range[0]:x_range[1]], "r", alpha=0.5)
plt.plot(data["control"][x_range[0]:x_range[1]], "blue", alpha=0.5)
plt.legend(["load", "control"])

plt.tight_layout()
plt.savefig("allostasis.pdf")
