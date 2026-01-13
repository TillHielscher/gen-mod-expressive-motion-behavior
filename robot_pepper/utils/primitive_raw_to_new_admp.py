import os
import pickle
#import peppertoolbox  # Custom module
import animation_dmp  # Custom module
import numpy as np
import json
import random

stand_init_joint_anges = {"HeadPitch": -10.0, "HipPitch": -2.0, "HipRoll": 0.0, "KneePitch": 0.0, "LElbowRoll": -30.0, "LElbowYaw": -70.0, "LHand": 0.0, "LShoulderPitch": 90.0, "LShoulderRoll": 10.0, "LWristYaw": 0.0, "RElbowRoll": 30.0, "RElbowYaw": 70.0, "RHand": 0.0, "RShoulderPitch": 90.0, "RShoulderRoll": -10.0, "RWristYaw": 0.0, "HeadYaw": 0.0}

def get_joint_idx(joint_name):
    if joint_name == "KneePitch":
        return 0
    elif joint_name == "HipPitch":
        return 1
    elif joint_name == "HipRoll":
        return 2
    elif joint_name == "HeadYaw":
        return 3
    elif joint_name == "HeadPitch":
        return 4
    elif joint_name == "LShoulderPitch":
        return 5
    elif joint_name == "LShoulderRoll":
        return 6
    elif joint_name == "LElbowYaw":
        return 7
    elif joint_name == "LElbowRoll":
        return 8
    elif joint_name == "LWristYaw":
        return 9
    elif joint_name == "LHand":
        return 10
    elif joint_name == "LFinger21":
        return 11
    elif joint_name == "LFinger22":
        return 12
    elif joint_name == "LFinger23":
        return 13
    elif joint_name == "LFinger11":
        return 14
    elif joint_name == "LFinger12":
        return 15
    elif joint_name == "LFinger13":
        return 16
    elif joint_name == "LFinger41":
        return 17
    elif joint_name == "LFinger42":
        return 18
    elif joint_name == "LFinger43":
        return 19
    elif joint_name == "LFinger31":
        return 20
    elif joint_name == "LFinger32":
        return 21
    elif joint_name == "LFinger33":
        return 22
    elif joint_name == "LThumb1":
        return 23
    elif joint_name == "LThumb2":
        return 24
    elif joint_name == "RShoulderPitch":
        return 25
    elif joint_name == "RShoulderRoll":
        return 26
    elif joint_name == "RElbowYaw":
        return 27
    elif joint_name == "RElbowRoll":
        return 28
    elif joint_name == "RWristYaw":
        return 29
    elif joint_name == "RHand":
        return 30
    elif joint_name == "RFinger41":
        return 31
    elif joint_name == "RFinger42":
        return 32
    elif joint_name == "RFinger43":
        return 33
    elif joint_name == "RFinger31":
        return 34
    elif joint_name == "RFinger32":
        return 35
    elif joint_name == "RFinger33":
        return 36
    elif joint_name == "RFinger21":
        return 37
    elif joint_name == "RFinger22":
        return 38
    elif joint_name == "RFinger23":
        return 39
    elif joint_name == "RFinger11":
        return 40
    elif joint_name == "RFinger12":
        return 41
    elif joint_name == "RFinger13":
        return 42
    elif joint_name == "RThumb1":
        return 43
    elif joint_name == "RThumb2":
        return 44
    elif joint_name == "WheelFL":
        return 45
    elif joint_name == "WheelB":
        return 46
    elif joint_name == "WheelFR":
        return 47
    else:
        raise(Exception("not a joint name" +  str(joint_name)))

def translate_traj_pepper_edmp(pepper_traj, use_zero = False):

    # initialize with zeros
    edmp_traj = np.zeros((len(list(pepper_traj[list(pepper_traj.keys())[0]])), 48))
    
    # iterate time steps
    for i in range(len(list(pepper_traj[list(pepper_traj.keys())[0]]))):
        
        # if default should not be zero values but instead stand init values
        if not use_zero:
            for key in stand_init_joint_anges.keys():
                edmp_traj[i,get_joint_idx(key)] = np.deg2rad(stand_init_joint_anges[key])

        # iterate keys (overwrite stand init or zero init)
        for key in pepper_traj.keys():
            
            if key=="Unnamed: 0" or len(key) <= 3:
                continue

            edmp_traj[i,get_joint_idx(key)] = np.array(list(pepper_traj[key]))[i]

    return edmp_traj

def load_json_from_folder(folder_path, file_selector):
    """
    Load a JSON object from a file in the specified folder.

    Parameters:
        folder_path (str): Path to the folder containing JSON files.
        file_selector (str): Either "random" or a descriptive string for matching.

    Returns:
        dict: The JSON object loaded from the selected file.
    """
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder '{folder_path}' does not exist.")

    # List all JSON files in the folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    if not json_files:
        raise ValueError("No JSON files found in the specified folder.")

    if file_selector.lower() == "random":
        # Randomly pick a file
        chosen_file = random.choice(json_files)
    else:
        # Scoring logic: prioritize connected substrings and shorter names
        file_selector_lower = file_selector.lower()
        matching_scores = {}

        for file_name in json_files:
            file_name_lower = file_name.lower()
            
            # Initialize score
            score = 0

            # Score based on connected substring matches
            for i in range(len(file_selector_lower)):
                for j in range(i + 1, len(file_selector_lower) + 1):
                    substring = file_selector_lower[i:j]
                    if substring in file_name_lower:
                        score += len(substring) ** 2  # Reward longer matches

            # Additional loose scoring (individual letter overlap)
            unique_selector_chars = set(file_selector_lower)
            unique_file_chars = set(file_name_lower)
            loose_score = len(unique_selector_chars & unique_file_chars)
            score += loose_score  # Add to total score

            # Apply a penalty for longer filenames
            length_penalty = len(file_name_lower) * 0.1
            score -= length_penalty

            # Store the score
            matching_scores[file_name] = score

        # Pick the file with the highest score
        chosen_file = max(matching_scores, key=matching_scores.get)

    # Construct the full path to the chosen file
    chosen_file_path = os.path.join(folder_path, chosen_file)

    print(chosen_file_path)

    # Load and return the JSON object
    with open(chosen_file_path, 'r') as file:
        traj_dict = json.load(file)

  # Convert the angles to radians
    angles_in_radians = {key: np.deg2rad(values) for key, values in traj_dict.items()}

    # Convert arrays back to lists for better readability (optional)
    angles_in_radians = {key: list(values) for key, values in angles_in_radians.items()}
    return angles_in_radians


# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths to the data directories
#gesture_lib_path = os.path.join(script_dir, "..", "primitive_raw_data_choreographe")
gesture_lib_path = os.path.join(script_dir, "..", "out_json")
pickle_output_dir = os.path.join(script_dir, "..", "primitive_saved")

# Create output directory if it doesn't exist
os.makedirs(pickle_output_dir, exist_ok=True)

# Check the contents of primitive_raw_data
print(f"Looking in {gesture_lib_path} for .json files...")

# Ensure the path exists
if not os.path.exists(gesture_lib_path):
    print(f"ERROR: The directory {gesture_lib_path} does not exist.")
else:
    # Collect all .json files in the given path
    gestures = [f for f in os.listdir(gesture_lib_path) 
                if f.endswith(".json")]

    # Debugging output
    print(f"Found the following .json files: {gestures}")

# If there are no gestures, exit early
if not gestures:
    print("No .json gesture files found. Exiting.")
    exit()

# Initialize a simple counter
total_gestures = len(gestures)

# Process each gesture in a single loop with a counter
for idx, gesture_file in enumerate(gestures, start=1):
    # Print progress
    print(f"Processing gesture {idx}/{total_gestures}: {gesture_file}")

    # Define full path to the .json file
    gesture_file_path = os.path.join(gesture_lib_path, gesture_file)

    # Load trajectory from the .json file
    gesture_traj = load_json_from_folder(gesture_lib_path, gesture_file)


    # Convert to EDMP format
    edmp_traj = translate_traj_pepper_edmp(gesture_traj)


    # Create DMP object    
    dmp = animation_dmp.DMP(edmp_traj, 100)

    # Save the object using the gesture file name (without the .json extension)
    output_filename = f"{os.path.splitext(gesture_file)[0]}"
    output_path = os.path.join(pickle_output_dir, output_filename)
    dmp.save(output_path)
    #with open(output_path, 'wb') as f:
    #    pickle.dump(dmp, f, pickle.HIGHEST_PROTOCOL)

print("Save files have been successfully saved.")
