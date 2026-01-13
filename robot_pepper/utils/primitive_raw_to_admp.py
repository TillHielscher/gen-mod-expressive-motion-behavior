import os
import pickle
import peppertoolbox  # Custom module
import animation_dmp  # Custom module

# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define absolute paths to the data directories
#gesture_lib_path = os.path.join(script_dir, "..", "primitive_raw_data_choreographe")
gesture_lib_path = os.path.join(script_dir, "..", "primitive_raw_data_test")
pickle_output_dir = os.path.join(script_dir, "..", "primitive_admp_picked")

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
    gesture_traj = peppertoolbox.load_json_from_folder(gesture_lib_path, gesture_file)


    # Convert to EDMP format
    edmp_traj = peppertoolbox.translate_traj_pepper_edmp(gesture_traj)


    # Create DMP object    
    dmp = animation_dmp.DMP(edmp_traj, 100)

    # Save the object using the gesture file name (without the .json extension)
    output_filename = f"{os.path.splitext(gesture_file)[0]}.pickle"
    output_path = os.path.join(pickle_output_dir, output_filename)
    with open(output_path, 'wb') as f:
        pickle.dump(dmp, f, pickle.HIGHEST_PROTOCOL)

print("Pickle files have been successfully saved.")
