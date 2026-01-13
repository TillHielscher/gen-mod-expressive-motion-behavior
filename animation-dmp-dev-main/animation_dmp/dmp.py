import numpy as np
from copy import deepcopy
from .phase import Phase
from .principles import *


class DMP():
    def __init__(self,
                 given_demo,
                 n_weights_dim,
                 alpha=50,
                 dt=1/60,
                 ) -> None:
        demo = deepcopy(given_demo)

        # Parameters
        self.n_weights_dim = n_weights_dim
        self.alpha = alpha
        self.beta = self.alpha / 4

        # Important properties
        self.dt = dt
        # Creation of the canonical system with the specified phase type
        self.phase = Phase()

        # Parameters inherited from the demo
        if demo.ndim == 1:
            demo = np.atleast_2d(demo).T
        self.n_dim = demo.shape[1]
        self.dim_priorities = self.analyse_demo(demo)
        self.start = demo[0, :]
        self.goal_original = demo[-1, :]
        self.goal = deepcopy(self.goal_original)
        self.tau_original = demo.shape[0] * self.dt
        self.tau = deepcopy(self.tau_original)
        self.w_original = self.imitate(demo)
        self.w = deepcopy(self.w_original)
        self.w_arc = deepcopy(self.w_original)
        self.w_rand = deepcopy(self.w_original)

        # Prepare dmo object to for animation modulation
        default_principle_parameters = get_default_principle_parameters()
        for key, value in default_principle_parameters.items():
            setattr(self, key, value)

    def set_principle_parameters(self,  **kwargs):
        """
        Set principle parameters. Can take the following arguments (and values): \n
        p_arc: float > 0, "reset" => 0.0\n
        p_ant: float > 0, "reset" => 0.0\n
        p_ant_t: float > 0, "reset" => 0.075\n
        p_ant_n: int > 0, "reset" => 1\n
        p_slow: bool, "reset" => False\n
        p_time: float > 0, "reset" => 1.0\n
        p_progression: list containingat least 3 of "slow" or "moderate" or "fast", "reset" => ["moderate", "moderate", "moderate"]\n
        p_exa: float > 0, "reset" => 1.0\n
        p_sec: float > 0, "reset" => 0.0\n
        p_sec_data: list of np.array([target_dim, source_dim, inverted (, direction_limiter)]) where target_dim and source_dim are integers and inverted and the optional direction_limiter are 1 or -1, "reset" = []\n
        p_follow: float > 0, "reset" => 0.0\n
        p_follow_data list of np.array([target_dim, source_dim, inverted (, condition_dim, lower_limit, upper_limit)]) where target_dim and source_dim and the optional condition_dim are integers and inverted is 1 or -1 and lower_limit and upper_limit (of the condition_dim) are floats, "reset" = []\n
        p_rand: float > 0, "reset" => 0.0\n
        p_goal= np array of size n_dim, "reset" => goal_original\n
        """
        set_principle_parameters(dmp_obj=self,
                                 **kwargs)

    # Inizialize the default state at the start of the demo and reset the time
    def init_state(self):
        self.y = deepcopy(self.start)
        self.yd = np.zeros(self.n_dim)
        self.ydd = np.zeros(self.n_dim)

        self.t = 0.0
        self.x = 1.0

    # Computation of the basis function for a given phase x and weight i
    def basis_function(self, x, i):
        # Calculate center if referenced weight i is the last one
        if i == self.n_weights_dim-1:
            c_i1 = 1 - (i / (self.n_weights_dim - 1))
            c_i = 1 - ((i-1) / (self.n_weights_dim - 1))
        # Calculate center for all other referenced weights i
        else:
            c_i1 = 1 - ((i+1) / (self.n_weights_dim - 1))
            c_i = 1 - (i / (self.n_weights_dim - 1))
        # Calculate basis function width
        h = (1/((c_i1-c_i)**2))
        # Calculate the basis function at center c_i and with width h
        psi = np.exp(-h * ((x-c_i)**2))
        return psi

    # Computation of the forcing term for a given phase x

    def forcing_term(self, x):
        # Generate basis functions
        psi = np.empty(self.n_weights_dim)
        for i in range(self.n_weights_dim):
            psi[i] = self.basis_function(x, i)

        # Compute f for each dimension
        f = np.empty(self.n_dim)
        for dim in range(self.n_dim):
            nom = 0
            denom = 0
            # Precalculation of the basis functions with the respective weights obtained from demo imitation
            for i in range(self.n_weights_dim):
                nom += psi[i] * self.w[dim, i]
                denom += psi[i]
            f[dim] = (nom / (denom+1e-10)) * x

        return f

    # Move the system state a single step (using time step dt) and calculate new state
    def step(self):
        # Calculate the current phase variable value
        self.x = self.phase.phase(self.t, self.tau)

        # Calculate the forcing term values
        f = self.forcing_term(self.x)

        # Apply exa
        f = apply_principle_exa(f, self.p_exa)

        # Calculate acceleration
        self.ydd = (1/self.tau ** 2)*((self.alpha * (self.beta *
                                                     (self.goal - self.y) - self.tau * self.yd)) + f)

        self.ydd = apply_principle_ant(
            self.ydd, self.p_ant, self.p_ant_t, self.p_ant_n, self.dim_priorities, self.t, self.tau)

        self.ydd = apply_principle_follow(
            self.ydd, self.y, self.p_follow, self.p_follow_data)

        # Euler integration to obtain velocities
        self.yd += self.dt * self.ydd

        # Euler integration to obtain positions
        self.y += self.dt * self.yd

        self.y = apply_principle_sec(
            self.y, self.yd, self.p_sec, self.p_sec_data)

        # Progress in time
        self.t += self.dt

    # Compute a whole run of the system and return the system trajectory and phase
    def run(self):
        # Initialize default state
        self.init_state()

        # Initialize datastructures for the runs trajectory and phase
        traj = []

        # Step until the execution time is reached
        while self.t < self.tau and abs(self.t-self.tau) > self.dt/2:
            self.step()
            traj.append(deepcopy(self.y))
        # Return an array of states and an array of phase variables for each step
        return np.array(traj)
    
    def get_state(self):
        state={}

        state["y"]=deepcopy(self.y)
        state["yd"]=deepcopy(self.yd)
        state["ydd"]=deepcopy(self.ydd)
        state["t"]=deepcopy(self.t)
        state["tau"]=deepcopy(self.tau)
        state["x"]=deepcopy(self.x)

        return state

    # Imitate a given demo to obtain weights
    def imitate(self, y_demo):
        # Compute demo velocities
        yd_demo = np.empty_like(y_demo)
        for dim in range(self.n_dim):
            yd_demo[:, dim] = np.gradient(y_demo[:, dim]) / self.dt
        yd_demo[-1, :] = 0.0

        # Compute demo accelerations
        ydd_demo = np.empty_like(y_demo)
        for dim in range(self.n_dim):
            ydd_demo[:, dim] = np.gradient(yd_demo[:, dim]) / self.dt
        ydd_demo[-1, :] = 0.0

        # Compute the required forcing term values given the demo data
        f_hat = (self.tau ** 2) * ydd_demo - self.alpha * \
            (self.beta * (self.goal - y_demo) - self.tau * yd_demo)

        # Initialize the weights
        w = np.empty((self.n_dim, self.n_weights_dim))

        # Compute the weights for each dimension
        for dim in range(self.n_dim):
            t = 0.0
            s = np.empty((y_demo.shape[0], 1))
            psi = np.empty((y_demo.shape[0], self.n_weights_dim))
            counter = 0

            # Iterate over the time window to create basis functions for a linear phase
            while t < self.tau and abs(t-self.tau) > self.dt/2:
                s[counter] = self.phase.phase_linear(
                    t, self.tau)
                for i in range(self.n_weights_dim):
                    psi[counter, i] = np.squeeze(
                        self.basis_function(s[counter], i))
                t += self.dt
                counter += 1

            # Compute the weight for each basis function
            for i in range(self.n_weights_dim):
                gamma = np.diag(psi[:, i])
                nom = np.squeeze(
                    np.matmul(s.T, np.matmul(gamma, f_hat[:, dim])))
                denom = np.squeeze(np.matmul(s.T, np.matmul(gamma, s)))
                w[dim, i] = nom / (denom+1e-10)
        return w

    # Compute the dimension priorities by analyzing the demo

    def analyse_demo(self, demo):
        # If the system is one dimensional just return a 1 dim array
        if self.n_dim == 1:
            return np.array([0])
        else:
            # Get the min and max values for each dimension and calculate the differences
            analysis_data = np.empty((self.n_dim, 3))  # min, max, diff
            for dim in range(self.n_dim):
                analysis_data[dim, 0] = min(demo[:, dim])
                analysis_data[dim, 1] = max(demo[:, dim])
                analysis_data[dim, 2] = abs(
                    analysis_data[dim, 1]-analysis_data[dim, 0])
            # Create a index array sorted by the differences
            highest_diff_dims = np.argsort(analysis_data[:, 2])[::-1]
            return highest_diff_dims

    # Manually move a dimension to a new priority in the dimension priorities

    def move_dim_to_priority(self, arr, dim, new_priority):
        new_priority = new_priority-1
        # Convert the NumPy array to a list
        arr_list = arr.tolist()

        # Check if the value exists in the list
        if dim not in arr_list:
            raise ValueError(f"Dimension {dim} not found in the array.")

        # Find the index of the value to move
        old_index = arr_list.index(dim)

        # Remove the value from its old position
        arr_list.pop(old_index)

        # Calculate the new index based on new priority
        # Ensure the new index is within the bounds of the list
        new_index = max(0, min(new_priority, len(arr_list)))

        # Insert the value at the new position
        arr_list.insert(new_index, dim)

        # Convert the list back to a NumPy array
        self.dim_priorities = np.array(arr_list)
