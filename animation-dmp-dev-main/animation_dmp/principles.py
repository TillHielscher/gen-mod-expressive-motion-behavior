
import numpy as np
from scipy.ndimage import gaussian_filter1d
from copy import deepcopy


def get_default_principle_parameters():
    default_principle_parameters = {}

    default_principle_parameters["p_arc"] = 0.0
    default_principle_parameters["p_ant"] = 0.0
    default_principle_parameters["p_ant_t"] = 0.075
    default_principle_parameters["p_ant_n"] = 1
    default_principle_parameters["p_slow"] = False
    default_principle_parameters["p_time"] = 1.0
    default_principle_parameters["p_progression"] = [
        "moderate", "moderate", "moderate"]
    default_principle_parameters["p_exa"] = 1.0
    default_principle_parameters["p_sec"] = 0.0
    default_principle_parameters["p_sec_data"] = []
    default_principle_parameters["p_follow"] = 0.0
    default_principle_parameters["p_follow_data"] = []
    default_principle_parameters["p_rand"] = 0.0

    return default_principle_parameters
    
def set_principle_parameters(dmp_obj, **kwargs):

    p_arc = kwargs.get("p_arc", None)
    p_ant = kwargs.get("p_ant", None)
    p_ant_t = kwargs.get("p_ant_t", None)
    p_ant_n = kwargs.get("p_ant_n", None)
    p_slow = kwargs.get("p_slow", None)
    p_time = kwargs.get("p_time", None)
    p_progression = kwargs.get("p_progression", None)
    p_exa = kwargs.get("p_exa", None)
    p_sec = kwargs.get("p_sec", None)
    p_sec_data = kwargs.get("p_sec_data", None)
    p_follow = kwargs.get("p_follow", None)
    p_follow_data = kwargs.get("p_follow_data", None)
    p_rand = kwargs.get("p_rand", None)
    p_goal = kwargs.get("p_goal", None)

    if p_arc is not None:
        if isinstance(p_arc, str) and p_arc == "reset":
            dmp_obj.p_arc = 0.0
            dmp_obj.w = deepcopy(dmp_obj.w_original)
        else:
            dmp_obj.p_arc = p_arc

            if dmp_obj.p_arc == 0.0:
                dmp_obj.w = deepcopy(dmp_obj.w_original)
            elif dmp_obj.p_arc > 0.0:
                dmp_obj.w = gaussian_filter1d(
                    dmp_obj.w_original, sigma=(1e-5 + dmp_obj.p_arc), axis=1)
            elif dmp_obj.p_arc < 0.0:
                w_smoothed = gaussian_filter1d(
                    dmp_obj.w_original, sigma=(1e-5 + -1*dmp_obj.p_arc), axis=1)
                dmp_obj.w = dmp_obj.w_original + \
                    (dmp_obj.w_original - w_smoothed)
            dmp_obj.w_arc = deepcopy(dmp_obj.w)

    if p_ant is not None:
        if isinstance(p_ant, str) and p_ant == "reset":
            dmp_obj.p_ant = 0.0
        else:
            dmp_obj.p_ant = p_ant

    if p_ant_t is not None:
        if isinstance(p_ant_t, str) and p_ant_t == "reset":
            dmp_obj.p_ant_t = 0.075
        else:
            dmp_obj.p_ant_t = p_ant_t

    if p_ant_n is not None:
        if isinstance(p_ant_n, str) and p_ant_n == "reset":
            dmp_obj.p_ant_n = 1
        else:
            dmp_obj.p_ant_n = p_ant_n

    if p_slow is not None:
        if isinstance(p_slow, str) and p_slow == "reset":
            dmp_obj.p_slow = False
            dmp_obj.phase.set_phase(p_slow=dmp_obj.p_slow)
        else:
            dmp_obj.p_slow = p_slow
            dmp_obj.phase.set_phase(p_slow=dmp_obj.p_slow)

    if p_time is not None:
        if isinstance(p_time, str) and p_time == "reset":
            dmp_obj.p_time = 1.0
            dmp_obj.tau = deepcopy(dmp_obj.tau_original)
        else:
            dmp_obj.p_time = p_time
            dmp_obj.tau = deepcopy(dmp_obj.tau_original) * dmp_obj.p_time

    if p_progression is not None:
        if isinstance(p_progression, str) and p_progression == "reset":
            dmp_obj.p_progression = ["moderate", "moderate", "moderate"]
            dmp_obj.phase.set_phase(
                progression_rates=dmp_obj.p_progression, p_slow=dmp_obj.p_slow)
        else:
            dmp_obj.p_progression = p_progression
            dmp_obj.phase.set_phase(
                progression_rates=dmp_obj.p_progression, p_slow=dmp_obj.p_slow)

    if p_exa is not None:
        if isinstance(p_exa, str) and p_exa == "reset":
            dmp_obj.p_exa = 1.0
        else:
            dmp_obj.p_exa = p_exa

    if p_sec is not None:
        if isinstance(p_sec, str) and p_sec == "reset":
            dmp_obj.p_sec = 0.0
        else:
            dmp_obj.p_sec = p_sec

    if p_sec_data is not None:
        if isinstance(p_sec, str) and p_sec == "reset":
            dmp_obj.p_sec_data = []
        else:
            dmp_obj.p_sec_data = p_sec_data

    if p_follow is not None:
        if isinstance(p_follow, str) and p_follow == "reset":
            dmp_obj.p_follow = 0.0
        else:
            dmp_obj.p_follow = p_follow

    if p_follow_data is not None:
        if isinstance(p_follow_data, str) and p_follow_data == "reset":
            dmp_obj.p_follow_data = []
        else:
            dmp_obj.p_follow_data = p_follow_data

    if p_rand is not None:
        if isinstance(p_rand, str) and p_rand == "reset":
            dmp_obj.p_rand = 0.0
            dmp_obj.w = deepcopy(dmp_obj.w_arc)
        else:
            dmp_obj.p_rand = p_rand
            if dmp_obj.p_rand > 0.0:
                for dim in range(dmp_obj.n_dim):
                    dmp_obj.w_rand[dim, :] = dmp_obj.w_arc[dim, :] + (1+np.mean(np.abs(
                        dmp_obj.w_arc[dim, :]))) * np.random.randn(dmp_obj.n_weights_dim) * dmp_obj.p_rand
                dmp_obj.w = deepcopy(dmp_obj.w_rand)

    if p_goal is not None:
        if isinstance(p_goal, str) and p_goal == "reset":
            dmp_obj.p_goal = deepcopy(dmp_obj.goal_original)
            dmp_obj.goal = dmp_obj.p_goal
        else:
            dmp_obj.p_goal = p_goal
            dmp_obj.goal = dmp_obj.p_goal


def apply_principle_ant(ydd, p_ant, p_ant_t, p_ant_n, dim_priorities, t, tau):
    if t/tau < p_ant_t and p_ant > 0:
        for n in range(p_ant_n):
            ydd[dim_priorities[n]] = - \
                ydd[dim_priorities[n]] * p_ant

    return ydd


def apply_principle_exa(f, p_exa):
    return f * p_exa


def apply_principle_sec(y, yd, p_sec, p_sec_data):
    if p_sec > 0 and len(p_sec_data) > 0:
        for relation in p_sec_data:
            target = int(relation[0])
            source = int(relation[1])
            if relation[2] == True:
                inverted = -1
            elif relation[2] == False:
                inverted = 1
            else:
                raise ValueError(
                    'The "inverted" tag must be either True or False.')
            # Only limit direction when value is given
            if len(relation) > 3:
                # negative values only apply sec for negative source vel:
                if relation[3] < 0:
                    # check vel is actually negative
                    if yd[source] < 0:
                        y[target] = y[target] + \
                            p_sec * yd[source] * inverted
                # positive values only apply sec for positive source vel:
                elif relation[3] > 0:
                    # check if vel is actually positive
                    if yd[source] > 0:
                        y[target] = y[target] + \
                            p_sec * yd[source] * inverted
                # if the specifier does not limit to a direction just execute sec
                else:
                    y[target] = y[target] + \
                        p_sec * yd[source] * inverted
            # No limitation specifier given
            else:
                y[target] = y[target] + \
                    p_sec * yd[source] * inverted

    return y


def apply_principle_follow(ydd, y, p_follow, p_follow_data):
    if p_follow > 0 and len(p_follow_data) > 0:
        for relation in p_follow_data:
            # Basic relation
            if len(relation) < 4:
                target = int(relation[0])
                source = int(relation[1])
                if relation[2] == True:
                    inverted = -1
                elif relation[2] == False:
                    inverted = 1
                else:
                    raise ValueError(
                        'The "inverted" tag must be either True or False.')
                ydd[target] = ydd[target] - \
                    p_follow * ydd[source] * inverted
            # Conditioned relation. Application only in specified ranges
            else:
                target = int(relation[0])
                source = int(relation[1])
                if relation[2] == True:
                    inverted = -1
                elif relation[2] == False:
                    inverted = 1
                else:
                    raise ValueError(
                        'The "inverted" tag must be either True or False.')
                condition_joint = int(relation[3])
                condition_lower = relation[4]
                condition_upper = relation[5]
                if y[condition_joint] > condition_lower and y[condition_joint] < condition_upper:
                    ydd[target] = ydd[target] - \
                        p_follow * \
                        ydd[source] * inverted
    return ydd
