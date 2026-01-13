# Animation DMP

A (rather basic) DMP inplementation which features modulation capabilities to animate system trajrectories.

# Installation

Clone the source

In your preferred environment navigate to the root of the project and install using pip

```
pip install .
```

# Usage

Basic usage given some reference trajectory (n_steps x n_dim) as demonstration.

```
import animation_dmp

dmp = animation_dmp.DMP(given_demo=demonstration, n_weights_dim=25)

trajectory, phase = dmp.run()
```

---

Animation modulation is possible with the following arguments:

p_arc: float > 0, "reset" => 0.0

p_ant: float > 0, "reset" => 0.0

p_ant_t: float > 0, "reset" => 0.075

p_ant_n: int > 0, "reset" => 1

p_slow: bool, "reset" => False

p_time: float > 0, "reset" => 1.0

p_progression: list containingat least 3 of "slow" or "moderate" or "fast", "reset" => ["moderate", "moderate", "moderate"]

p_exa: float > 0, "reset" => 1.0

p_sec: float > 0, "reset" => 0.0

p_sec_data: list of np.array([target_dim, source_dim, inverted (, direction_limiter)]) where target_dim and source_dim are integers and inverted and the optional direction_limiter are 1 or -1, "reset" = []

p_follow: float > 0, "reset" => 0.0

p_follow_data list of np.array([target_dim, source_dim, inverted (, condition_dim, lower_limit, upper_limit)]) where target_dim and source_dim and the optional condition_dim are integers and inverted is 1 or -1 and lower_limit and upper_limit (of the condition_dim) are floats, "reset" = []

p_rand: float > 0, "reset" => 0.0

p_goal= np array of size n_dim, "reset" => goal_original

```
import animation_dmp

dmp = animation_dmp.DMP(given_demo=demonstration, n_weights_dim=25)
dmp.set_principle_parameters(p_exa=1.5)

trajectory, phase = dmp.run()
```

---

Rollout can be done step wise:

```
import animation_dmp

dmp = animation_dmp.DMP(given_demo=demonstration, n_weights_dim=25)

dmp.init_state()
dmp.step()
state = dmp.get_state()
```

---

Further examples can be found in the examples directory.