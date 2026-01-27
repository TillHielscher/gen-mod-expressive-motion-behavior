import os
import pickle
import random
import yaml
from copy import deepcopy
import logging
import animation_dmp

class LLAGTimelineBlock:
    def __init__(self, name_identifier, context_params=None, primitive_path="robot_pepper_primitives", idle_data_yaml_path="robot_pepper/robot_data.yaml"):
        self.name_identifier = name_identifier
        self.dmp = self.load_save_from_folder(primitive_path, name_identifier, idle_yaml_path=idle_data_yaml_path)
        self.dmp.init_state()
        state = self.dmp.get_state()

    def step(self):
        self.dmp.step()

   # update pepper head pose based on simulated data
    def update_goal(self, rt_data, gain=0.01):
        logging.debug(f"[BLOCK GOAL] rt data: {rt_data}")
        
        block_goal = self.dmp.goal
        updated_goal = deepcopy(block_goal)

        #gain = 0.01  # Tune this as needed

        # Correct update: move toward the target by following the sign of the error
        yaw_delta = gain * rt_data['x']
        pitch_delta = gain * rt_data['y']

        updated_goal[3] += yaw_delta
        updated_goal[4] += pitch_delta

        logging.debug(f"[BLOCK GOAL] yaw += {yaw_delta}, pitch += {pitch_delta}")

        self.dmp.set_principle_parameters(p_goal=updated_goal)
        logging.debug(f"[BLOCK GOAL] updated goal: yaw={updated_goal[3]}, pitch={updated_goal[4]}")


    def set_context_params(self, context_params):
        pass

    def is_complete(self):
        state = self.dmp.get_state()
        if state["t"] >= state["tau"]:
            return True

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def load_save_from_folder(self, folder_path, name_identifier, idle_yaml_path=None):
        """
        Load a DMP object from serialized files in the specified folder.

        Parameters:
            folder_path (str): Path to the folder containing DMP files (.json + .npy).
            name_identifier (str): Either "random" or a descriptive string for matching.

        Returns:
            animation_dmp.DMP: The loaded DMP object.
        """


        chosen_file_base = None

        if not os.path.isdir(folder_path):
            raise ValueError(f"The folder '{folder_path}' does not exist.")
        

        # List all .json files (each DMP has a .json and .npy pair)
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]


        if not json_files:
            raise ValueError("No DMP files found in the specified folder.")
        

        # Remove .json extension to get base names
        base_names = [f[:-5] for f in json_files]


        if name_identifier.lower() == "random":
            chosen_file_base = random.choice(base_names)
        elif name_identifier.lower() == "idle":
            if not idle_yaml_path or not os.path.isfile(idle_yaml_path):
                raise ValueError("idle_yaml_path must be provided and point to a valid .yaml file.")

            with open(idle_yaml_path, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)

            idle_names = yaml_data.get("idle_lib", [])
            if not idle_names:
                raise ValueError("No 'primitive_lib_idle' list found in the YAML file.")

            name_identifier = random.choice(idle_names)
            self.name_identifier = name_identifier
            
        
        if name_identifier.lower() not in ["random", "idle"]:
            name_identifier_lower = name_identifier.lower()
            matching_scores = {}

            # try direct match first
            length = 0
            for base_name in base_names:                
                if name_identifier_lower == base_name.lower():                    
                    if len(name_identifier_lower) > length:
                        length = len(name_identifier_lower)
                        chosen_file_base = base_name
                    
            if chosen_file_base is not None:
                logging.info(f"Direct match found: {chosen_file_base}")
            
            if chosen_file_base is None:
                for base_name in base_names:
                    base_name_lower = base_name.lower()
                    score = 0

                    # Score based on connected substring matches
                    for i in range(len(name_identifier_lower)):
                        for j in range(i + 1, len(name_identifier_lower) + 1):
                            substring = name_identifier_lower[i:j]
                            if substring in base_name_lower:
                                score += len(substring) ** 2

                    # Additional loose character match
                    unique_id_chars = set(name_identifier_lower)
                    unique_file_chars = set(base_name_lower)
                    loose_score = len(unique_id_chars & unique_file_chars)
                    score += loose_score

                    # Length penalty
                    score -= len(base_name_lower) * 0.1

                    matching_scores[base_name] = score

                chosen_file_base = max(matching_scores, key=matching_scores.get)

        chosen_file_path = os.path.join(folder_path, chosen_file_base)
        logging.info(f"[block] Selected file: {chosen_file_path}")

        # Load the DMP using animation_dmp.DMP.load()
        loaded_object = animation_dmp.DMP.load(chosen_file_path)

        return loaded_object




