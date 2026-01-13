import numpy as np
from copy import deepcopy

class MultiDMP:
    def __init__(self, base_dmp):
        self.dmps = []
        self.dmps.append(base_dmp)
        self.n_dmps = 1
        self.dmp_dim = base_dmp.n_dim
        self.ydd = np.zeros(base_dmp.n_dim)
        self.yd = np.zeros(base_dmp.n_dim)
        self.y = deepcopy(base_dmp.start)
        self.t = 0.0
        self.dt = 1/60
    
    def add_dmp(self, dmp):
        self.dmps.append(dmp)
        self.n_dmps += 1

    def run(self):
        traj = []
        ydd = []

        all_tau = []        
        for dmp in self.dmps:
            all_tau.append(dmp.tau)
            dmp.init_state()

        while self.t < max(all_tau):
            dmps_ydd = []
            for dmp in self.dmps:
                if dmp.t < dmp.tau:
                    dmp.step()
                    state = dmp.get_state()
                    dmps_ydd.append(state["ydd"])
            dmps_ydd = np.array(dmps_ydd) #n_dmps x dmp_dim

            weights = np.abs(dmps_ydd)/(sum(dmps_ydd)+1e-5)
            weights = weights/sum(weights) #n_dmps x dmp_dim
            ydd_temp = np.zeros(self.dmp_dim)
            for i, step in enumerate(dmps_ydd):
                ydd_temp += step * weights[i]
            self.ydd = ydd_temp
            
            self.yd += self.dt * self.ydd
            self.y += self.dt * self.yd
            traj.append(deepcopy(self.y))
            ydd.append(deepcopy(self.ydd))
            

            self.t += self.dt

        return np.array(traj), np.array(ydd)