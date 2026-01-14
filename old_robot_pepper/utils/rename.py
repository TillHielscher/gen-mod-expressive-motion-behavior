import os
import shutil

# Paths (relative to the script location)
base_dir = os.path.dirname(os.path.abspath(__file__))
old_folder = os.path.join(base_dir, "..", "primitive_admp_picked_old")
new_folder = os.path.join(base_dir, "..", "primitive_admp_picked")

# Ensure target folder exists
os.makedirs(new_folder, exist_ok=True)

# Mapping from old_name -> new_name
name_mapping = {
    "Back_to_stand": "return_to_stand",
    "Settle_to_stand": "settle_to-stand",
    "Wave_arm": "wave_arm",
    "Humble_bow": "bow_humble",
    "Show_surroundings": "show_surroundings",
    "Complain_strong": "complain",
    "Calm_down": "calm_surroundings",
    "Make_space": "make_space",
    "Elaborate": "elaborate",
    "Decide_options": "decide_options",
    "Check_Floor_Left": "check_floor_left",
    "Check_Floor_Right": "check_floor_right",
    "Funny": "react_funny",
    "Sway_joy": "sway_joyful",
    "Disappointment": "show_disappointment",
    "Question": "question",
    "Attract": "invite_closer",
    "Point_Left": "point_left",
    "Point_Right": "point_right",
    "Indicate_Left": "indicate_left",
    "Indicate_Right": "indicate_right",
    "Bow_down": "bow_respectful",
    "Excited": "show_excitement",
    "Explain": "explain",
    "Disagree_head": "disagree_shake_head",
    "Disagree_hand": "disagree_dismissive_wave",
    "Celebrate": "celebrate",
    "Think": "think",
    "Point_Front": "point_forward",
    "Hands_on_hips": "hands_on_hips",
    "nod_agreement": "nod_agreement",
    "nod_confirmation": "nod_confirmation",
    "shrug": "shrug",
    "scratch_head": "scratch_head",
    "offer_item": "offer_item",
}

# Copy and rename files
for old_name, new_name in name_mapping.items():
    old_path = os.path.join(old_folder, f"{old_name}.pickle")
    new_path = os.path.join(new_folder, f"{new_name}.pickle")

    if os.path.exists(old_path):
        shutil.copy2(old_path, new_path)
        print(f"Copied: {old_name}.pickle -> {new_name}.pickle")
    else:
        print(f"⚠️ Missing file: {old_name}.pickle")

print("\n✅ Done.")
