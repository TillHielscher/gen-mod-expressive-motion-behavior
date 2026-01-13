import numpy as np
import json

def reformat_motion_data(names, times, keys, fps=60):
    # Find the global max time
    max_time = max(max(t) for t in times)
    dt = 1.0 / fps
    uniform_times = np.arange(0, max_time + dt/2, dt)  # timeline
    
    output = {}
    for joint_name, joint_times, joint_keys in zip(names, times, keys):
        joint_times = np.array(joint_times)
        joint_keys = np.array(joint_keys)
        # convert rad to deg
        joint_keys = np.degrees(joint_keys)
        
        # Interpolate values at uniform times
        interpolated = np.interp(uniform_times, joint_times, joint_keys)
        
        # Store as list
        output[joint_name] = interpolated.tolist()
    
    return output

# ---- Your input data ----
names = []
times = []
keys = []

# single nod
""" filename = "nod_confirmation"
names.append("HeadPitch")
times.append([0, 1.04, 1.96])
keys.append([-0.00153399, 0.387463, 0])

names.append("HeadYaw")
times.append([0])
keys.append([0.116583])

names.append("HipPitch")
times.append([0])
keys.append([0.0184078])

names.append("HipRoll")
times.append([0])
keys.append([-0.0199418])

names.append("KneePitch")
times.append([0])
keys.append([0.0613592])

names.append("LElbowRoll")
times.append([0])
keys.append([-0.401903])

names.append("LElbowYaw")
times.append([0])
keys.append([-1.21031])

names.append("LHand")
times.append([0])
keys.append([0.466608])

names.append("LShoulderPitch")
times.append([0])
keys.append([1.54932])

names.append("LShoulderRoll")
times.append([0])
keys.append([0.139592])

names.append("LWristYaw")
times.append([0])
keys.append([-0.154976])

names.append("RElbowRoll")
times.append([0])
keys.append([0.415709])

names.append("RElbowYaw")
times.append([0])
keys.append([1.21491])

names.append("RHand")
times.append([0])
keys.append([0.488576])

names.append("RShoulderPitch")
times.append([0])
keys.append([1.54932])

names.append("RShoulderRoll")
times.append([0])
keys.append([-0.15033])

names.append("RWristYaw")
times.append([0])
keys.append([0.15029]) """


# nod
""" filename = "nod_agreement"
names.append("HeadPitch")
times.append([0, 0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([0, 0.477068, -0.30066, 0.398835, -0.5044, 0.0107379])

names.append("HeadYaw")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([0.00920391, -0.0521553, -0.0444853, -0.0444853, -0.0444853])

names.append("HipPitch")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([-0.0107379, 0.0214758, 0.0322137, 0.0429513, 0.0429513])

names.append("HipRoll")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([-0.0199418, -0.0199418, -0.0199418, -0.0199418, -0.0199418])

names.append("KneePitch")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([0.0214758, 0.0644271, 0.0889709, 0.0889709, 0.0889709])

names.append("LElbowRoll")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([-0.400369, -0.398835, -0.523088, -0.50468, -0.444854])

names.append("LElbowYaw")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([-1.21031, -1.21031, -1.21031, -1.21031, -1.21031])

names.append("LHand")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([0.464851, 0.398067, 0.398067, 0.45167, 0.391037])

names.append("LShoulderPitch")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([1.55852, 1.56006, 1.5708, 1.56926, 1.56926])

names.append("LShoulderRoll")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([0.139592, 0.139592, 0.139592, 0.139592, 0.139592])

names.append("LWristYaw")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([-0.154976, -0.154976, -0.154976, -0.154976, -0.154976])

names.append("RElbowRoll")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([0.414175, 0.395767, 0.527689, 0.506214, 0.435651])

names.append("RElbowYaw")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([1.21491, 1.21491, 1.21491, 1.21491, 1.21491])

names.append("RHand")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([0.48594, 0.397188, 0.397188, 0.431459, 0.392794])

names.append("RShoulderPitch")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([1.55085, 1.55239, 1.56313, 1.56159, 1.56466])

names.append("RShoulderRoll")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([-0.15033, -0.15033, -0.15033, -0.15033, -0.15033])

names.append("RWristYaw")
times.append([0.92, 1.84, 2.88, 3.92, 4.76])
keys.append([0.161028, 0.15029, 0.15029, 0.15029, 0.15029]) """

# offer item
""" filename = "offer_item"
names.append("HeadPitch")
times.append([0, 0.4, 0.92, 1.52, 2.76, 3.44, 4.76])
keys.append([0.0291457, 0.0291457, 0.0907571, 0.197222, 0.200952, 0.205554, 0.0260777])

names.append("HeadYaw")
times.append([0, 0.4, 0.92, 1.52, 2.2, 2.76, 3.44, 4.76])
keys.append([0.111981, 0.111981, -0.00698132, -0.174533, -0.118682, -0.125787, -0.110447, 0.111981])

names.append("HipPitch")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([-0.121185, -0.121185, -0.107379, -0.107379, -0.107379, -0.107379])

names.append("HipRoll")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([-0.0168738, -0.0168738, -0.0168738, -0.0168738, -0.0168738, -0.0168738])

names.append("KneePitch")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([0.14266, 0.14266, 0.174874, 0.174874, 0.174874, 0.174874])

names.append("LElbowRoll")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([-0.513884, -0.513884, -0.513884, -0.513884, -0.513884, -0.513884])

names.append("LElbowYaw")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([-1.20417, -1.20417, -1.20417, -1.20417, -1.20417, -1.20417])

names.append("LHand")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([0.405097, 0.405097, 0.405097, 0.405097, 0.405097, 0.405097])

names.append("LShoulderPitch")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([1.60301, 1.60301, 1.59074, 1.59074, 1.59074, 1.59074])

names.append("LShoulderRoll")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([0.130388, 0.130388, 0.130388, 0.130388, 0.130388, 0.130388])

names.append("LWristYaw")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([-0.0521979, -0.0521979, -0.0521979, -0.0521979, -0.0521979, -0.0521979])

names.append("RElbowRoll")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([0.543029, 0.814544, 0.48934, 0.0905049, 0.0920389, 0.544563])

names.append("RElbowYaw")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([1.1873, 1.22565, 1.22105, 1.22105, 1.20878, 1.17963])

names.append("RHand")
times.append([0, 0.4, 1.52, 2.2, 2.76, 3.44, 4.76])
keys.append([0.407733, 0.407733, 0.407733, 0.98, 0.92355, 0.899824, 0.422671])

names.append("RShoulderPitch")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([1.58614, 1.07072, 0.524621, 0.125787, 0.139592, 1.56313])

names.append("RShoulderRoll")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([-0.128854, -0.0659611, -0.0245438, -0.00306797, -0.0184078, -0.121185])

names.append("RWristYaw")
times.append([0, 0.4, 1.52, 2.76, 3.44, 4.76])
keys.append([0.118076, 1.15353, 1.23023, 1.79167, 1.76252, 0.177902]) """

#scratch head
""" filename = "scratch_head"
names.append("HeadPitch")
times.append([0, 0.48, 0.96, 1.48, 1.96, 3.16, 3.64, 4.76, 5.32, 5.92])
keys.append([0.0214758, 0.0107379, 0.0107379, 0.366621, 0.382742, 0.445059, 0.280998, 0.164061, 0.0191986, 0.0276117])

names.append("HeadYaw")
times.append([0, 0.48, 0.96, 1.48, 1.96, 3.16, 4.76, 5.32, 5.92])
keys.append([0.121185, 0.131922, 0.131922, 0.234699, 0.354609, 0.382227, 0.0226893, -0.010472, 0.115049])

names.append("HipPitch")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([-0.139592, -0.139592, -0.139592, -0.128854, -0.128854, -0.118117])

names.append("HipRoll")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([-0.0153399, -0.00460196, -0.00460196, 0.00613594, 0.00613594, -0.0153399])

names.append("KneePitch")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([0.107379, 0.118117, 0.118117, 0.153398, 0.153398, 0.141126])

names.append("LElbowRoll")
times.append([0, 0.48, 0.96, 1.48, 1.96, 3.16, 3.64, 4.2, 5.92])
keys.append([-0.52002, -0.980214, -1.52938, -1.55546, -1.55546, -1.52193, -1.56207, -1.51669, -0.521554])

names.append("LElbowYaw")
times.append([0, 0.48, 0.96, 1.48, 1.96, 2.56, 3.16, 4.2, 5.92])
keys.append([-1.20417, -1.68585, -1.85458, -1.55239, -1.55239, -1.35263, -1.33692, -1.37881, -1.20264])

names.append("LHand")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([0.393673, 0.393673, 0.393673, 0.393673, 0.393673, 0.406854])

names.append("LShoulderPitch")
times.append([0, 0.48, 0.96, 1.48, 1.96, 2.56, 3.16, 3.64, 4.2, 5.92])
keys.append([1.5892, 1.35911, 0.941864, 0.543029, -0.202458, -0.408407, 0.0994838, -0.459022, -0.164061, 1.59841])

names.append("LShoulderRoll")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([0.130388, 0.141126, 0.282253, 0.239301, 0.239301, 0.122719])

names.append("LWristYaw")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([-0.09515, -0.423426, -0.368202, -0.633584, -0.633584, -0.0353239])

names.append("RElbowRoll")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([0.536893, 0.536893, 0.536893, 0.15033, 0.161068, 0.543029])

names.append("RElbowYaw")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([1.19497, 1.19497, 1.19497, 1.19344, 1.19344, 1.18884])

names.append("RHand")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([0.396309, 0.396309, 0.396309, 0.396309, 0.396309, 0.408612])

names.append("RShoulderPitch")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([1.58, 1.58, 1.58, 1.56159, 1.56159, 1.59687])

names.append("RShoulderRoll")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([-0.133456, -0.133456, -0.133456, -0.251573, -0.251573, -0.127321])

names.append("RWristYaw")
times.append([0, 0.48, 0.96, 1.48, 1.96, 5.92])
keys.append([0.124212, 0.124212, 0.124212, 0.194776, 0.194776, 0.11961]) """


# shrug
filename = "shrug"
names.append("HeadPitch")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.0122719, 0.0214758, -0.146608, -0.10821, -0.111981, 0.123918, 0.0352817, 0.0245438])

names.append("HeadYaw")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.125787, 0.128854, 0.0663225, 0.010472, 0.0230098, 0.293215, 0.128854, 0.131922])

names.append("HipPitch")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([-0.15708, -0.147262, -0.147262, -0.147262, -0.144194, -0.174874, -0.174874, -0.153398])

names.append("HipRoll")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.00153399, 0.00153399, 0.146608, 0.13439, 0.115049, 0.0418879, 0.00153399, 0.0122719])

names.append("KneePitch")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.082835, 0.116583, 0.116583, 0.116583, 0.139592, 0.0720971, 0.0935729, 0.13499])

names.append("LElbowRoll")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([-0.524621, -0.661146, -0.814544, -1.4895, -1.48029, -0.317534, -0.139592, -0.52002])

names.append("LElbowYaw")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([-1.20724, -1.90214, -2.08621, -2.08775, -2.07701, -2.07087, -1.67204, -1.21491])

names.append("LHand")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.38225, 0.394552, 0.394552, 0.394552, 0.394552, 0.394552, 0.394552, 0.394552])

names.append("LShoulderPitch")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([1.5754, 1.24866, 1.19651, 1.14895, 1.15355, 1.05078, 1.37598, 1.57847])

names.append("LShoulderRoll")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.133456, 0.111981, 0.0598252, 0.0628932, 0.0674951, 0.102777, 0.0674951, 0.125787])

names.append("LWristYaw")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([-0.14117, -0.257754, -0.89283, -0.9772, -0.937316, -0.287979, -0.0174533, -0.190258])

names.append("RElbowRoll")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.529223, 0.859029, 1.00169, 1.42507, 1.42047, 0.418777, 0.240835, 0.535359])

names.append("RElbowYaw")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([1.20724, 1.83771, 2.09082, 2.10002, 2.09082, 2.04633, 1.32689, 1.19957])

names.append("RHand")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.383128, 0.394552, 0.394552, 0.394552, 0.394552, 0.394552, 0.394552, 0.394552])

names.append("RShoulderPitch")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([1.57233, 1.24713, 1.21951, 1.21798, 1.23025, 1.14588, 1.4128, 1.5708])

names.append("RShoulderRoll")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([-0.130388, 0, 0.00460196, 0.00613594, -0.00920391, -0.0199418, -0.0337477, -0.124253])

names.append("RWristYaw")
times.append([0, 0.48, 0.96, 1.44, 1.84, 2.4, 3.36, 3.96])
keys.append([0.145688, 0.443284, 0.977116, 1.13052, 1.0891, 0.0488692, 0.165806, 0.187106])



# ---- Process and save ----
motion_dict = reformat_motion_data(names, times, keys)

# Save as JSON
with open(filename +".json", "w") as f:
    json.dump(motion_dict, f, indent=2)

print("Saved motion data to motion_data.json")
