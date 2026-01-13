import animation_dmp

import numpy as np
import matplotlib.pyplot as plt

ydata = np.loadtxt("ydata.txt")

demo = ydata
dmp = animation_dmp.DMP(demo, 25)
demo, phase = dmp.run()

dmp1 = animation_dmp.DMP(demo, 25)
dmp1.set_principle_parameters(p_exa=1.5)

traj1, phase1 = dmp1.run()

fig, ax = plt.subplots()
ax.plot(demo, 'k--', linewidth=10)
ax.plot(traj1, 'r', linewidth=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines["left"].set_position(("data", 0))
ax.spines["bottom"].set_position(("data", 0))

ax.spines["left"].set_linewidth(10)
ax.spines["bottom"].set_linewidth(10)
ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(),
        clip_on=False, markersize=30)
ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(),
        clip_on=False, markersize=30)
# for major ticks
ax.set_xticks([])
ax.set_yticks([])
# for minor ticks
ax.set_xticks([], minor=True)
ax.set_yticks([], minor=True)
ax.set_xlim(left=0)
ax.set_ylabel("$y$", fontsize=75, rotation=0, labelpad=25)
ax.set_xlabel("$n_t$", fontsize=75)
ax.set_xlim(left=0)
fig.set_size_inches(20, 8)
plt.show()