import animation_dmp

import numpy as np
import matplotlib.pyplot as plt

linear_data = np.linspace(0,-3, 600)
custom_data = np.loadtxt("ydata.txt")

data1 = np.array([custom_data, custom_data])
data1 = np.transpose(data1)

data2 = np.array([linear_data, -1*(custom_data)])
data2 = np.transpose(data2)

data3 = np.array([np.zeros(600), np.zeros(600)])
data3 = np.transpose(data3)

dmp1 = animation_dmp.DMP(data1, 25)
dmp1.init_state()
state = dmp1.get_state()
traj1 = []
ydd1 = []
while state["t"] < state["tau"]:
    dmp1.step()
    state = dmp1.get_state()
    traj1.append(state["y"])
    ydd1.append(state["ydd"])
traj1 = np.array(traj1)
ydd1 = np.array(ydd1)
#print(traj1)

dmp2 = animation_dmp.DMP(data2, 25)
dmp2.init_state()
state = dmp2.get_state()
traj2 = []
ydd2 = []
while state["t"] < state["tau"]:
    dmp2.step()
    state = dmp2.get_state()
    traj2.append(state["y"])
    ydd2.append(state["ydd"])
traj2 = np.array(traj2)
ydd2 = np.array(ydd2)
#print(traj2)

dmp3 = animation_dmp.DMP(data3, 25)
dmp3.init_state()
state = dmp3.get_state()
traj3 = []
ydd3 = []
while state["t"] < state["tau"]:
    dmp3.step()
    state = dmp3.get_state()
    traj3.append(state["y"])
    ydd3.append(state["ydd"])
traj3 = np.array(traj3)
ydd3 = np.array(ydd3)
#print(traj3)

dmpmulti2 = animation_dmp.MultiDMP(dmp1)
dmpmulti2.add_dmp(dmp2)
trajmulti2, yddmulti2 = dmpmulti2.run()

dmpmulti = animation_dmp.MultiDMP(dmp1)
dmpmulti.add_dmp(dmp2)
dmpmulti.add_dmp(dmp3)
trajmulti, yddmulti = dmpmulti.run()
#print(trajmulti)


#print(trajmulti2)

fig, ax = plt.subplots(5, 3)
ax[0,0].plot(data1[:,0], 'k--', linewidth=10, label="data1 dim 0")
ax[0,0].plot(traj1[:,0], 'r', linewidth=8, label="traj1 dim 0")
ax[0,1].plot(data1[:,1], 'k--', linewidth=10, label="data1 dim 1")
ax[0,1].plot(traj1[:,1], 'r', linewidth=8, label="traj1 dim 1")
ax[0,2].plot(ydd1[:,0], 'r', linewidth=8, label="ydd1 dim 0")
ax[0,2].plot(ydd1[:,1], 'b', linewidth=8, label="ydd1 dim 1")


ax[1,0].plot(data2[:,0], 'k--', linewidth=10, label="data2 dim 0")
ax[1,0].plot(traj2[:,0], 'r', linewidth=8, label="traj2 dim 0")
ax[1,1].plot(data2[:,1], 'k--', linewidth=10, label="data2 dim 1")
ax[1,1].plot(traj2[:,1], 'r', linewidth=8, label="traj2 dim 1")
ax[1,2].plot(ydd2[:,0], 'r', linewidth=8, label="ydd2 dim 0")
ax[1,2].plot(ydd2[:,1], 'b', linewidth=8, label="ydd2 dim 1")

ax[2,0].plot(data3[:,0], 'k--', linewidth=10, label="data2 dim 0")
ax[2,0].plot(traj3[:,0], 'r', linewidth=8, label="traj2 dim 0")
ax[2,1].plot(data3[:,1], 'k--', linewidth=10, label="data2 dim 1")
ax[2,1].plot(traj3[:,1], 'r', linewidth=8, label="traj2 dim 1")
ax[2,2].plot(ydd3[:,0], 'r', linewidth=8, label="ydd2 dim 0")
ax[2,2].plot(ydd3[:,1], 'b', linewidth=8, label="ydd2 dim 1")


ax[3,0].plot(trajmulti[:,0], 'r', linewidth=8, label="trajmulti dim 0")
ax[3,1].plot(trajmulti[:,1], 'r', linewidth=8, label="trajmulti dim 1")
ax[3,0].plot(trajmulti2[:,0], 'b', linewidth=8, label="trajmulti dim 0")
ax[3,1].plot(trajmulti2[:,1], 'b', linewidth=8, label="trajmulti dim 1")
ax[3,2].plot(yddmulti[:,0], 'r', linewidth=8, label="yddmulti dim 0")
ax[3,2].plot(yddmulti[:,1], 'b', linewidth=8, label="yddmulti dim 1")

""" ax[4,0].plot(trajmulti2[:,0], 'r', linewidth=8, label="trajmulti dim 0")
ax[4,1].plot(trajmulti2[:,1], 'r', linewidth=8, label="trajmulti dim 1")
ax[4,2].plot(yddmulti2[:,0], 'r', linewidth=8, label="yddmulti dim 0")
ax[4,2].plot(yddmulti2[:,1], 'b', linewidth=8, label="yddmulti dim 1") """

ax[0,0].set_title('traj1 dim 0')
ax[0,1].set_title('traj1 dim 1')
ax[1,0].set_title('traj2 dim 0')
ax[1,1].set_title('traj2 dim 1')
ax[2,0].set_title('trajmulti dim 0')
ax[2,1].set_title('trajmulti dim 1')

#fig.set_size_inches(20, 8)

plt.show()

""" demo = ydata
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
plt.show() """