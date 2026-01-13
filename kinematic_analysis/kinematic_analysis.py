import xml.etree.ElementTree as ET
import numpy as np
from math import cos, sin
from collections import defaultdict


def rpy_to_matrix(r, p, y):
    """Convert roll-pitch-yaw to rotation matrix."""
    Rz = np.array([[cos(y), -sin(y), 0],
                   [sin(y),  cos(y), 0],
                   [0,       0,      1]])
    Ry = np.array([[cos(p), 0, sin(p)],
                   [0,       1, 0],
                   [-sin(p), 0, cos(p)]])
    Rx = np.array([[1, 0, 0],
                   [0, cos(r), -sin(r)],
                   [0, sin(r),  cos(r)]])
    return Rz @ Ry @ Rx


def axis_angle_matrix(axis, theta):
    """Rodrigues formula for rotation about an axis."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = cos(theta)
    s = sin(theta)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
    ])


class URDFKinematicAnalyzer:
    def __init__(self, urdf_path):
        self.joints = {}
        self.child_map = defaultdict(list)
        self.parent_map = {}
        self._parse_urdf(urdf_path)

    def _parse_urdf(self, urdf_path):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint in root.findall("joint"):
            jname = joint.attrib["name"]
            jtype = joint.attrib["type"]
            if jtype not in ["revolute"]:
                continue

            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]

            origin_tag = joint.find("origin")
            rpy = [float(x) for x in origin_tag.attrib.get("rpy", "0 0 0").split()] if origin_tag is not None else [0, 0, 0]

            axis_tag = joint.find("axis")
            axis = np.array([float(x) for x in axis_tag.attrib.get("xyz", "0 0 1").split()])
            axis = axis / np.linalg.norm(axis)

            limit_tag = joint.find("limit")
            lower = float(limit_tag.attrib.get("lower", -np.pi)) if limit_tag is not None else -np.pi
            upper = float(limit_tag.attrib.get("upper", np.pi)) if limit_tag is not None else np.pi

            self.joints[jname] = {
                "type": jtype,
                "parent": parent,
                "child": child,
                "axis_local": axis,
                "rpy": rpy,
                "lower": lower,
                "upper": upper
            }

            self.child_map[parent].append(jname)
            self.parent_map[child] = jname

    def forward_axis(self, joint_name, joint_values):
        """Compute global axis of a joint given joint values (dict: joint_name -> angle)."""
        chain = []
        cur_joint = joint_name
        while cur_joint in self.joints:
            chain.append(cur_joint)
            parent_link = self.joints[cur_joint]["parent"]
            if parent_link not in self.parent_map:
                break
            cur_joint = self.parent_map[parent_link]
        chain = chain[::-1]  # root to joint

        T = np.eye(3)
        for j in chain:
            info = self.joints[j]
            R_origin = rpy_to_matrix(*info["rpy"])
            T = T @ R_origin
            if info["type"] == "revolute":
                theta = joint_values.get(j, 0.0)
                R_joint = axis_angle_matrix(info["axis_local"], theta)
                T = T @ R_joint
        axis_global = T @ self.joints[joint_name]["axis_local"]
        return axis_global / np.linalg.norm(axis_global)

    def find_relations(self, samples=100, tol=0.95):
        relations = []

        # 1. Check conditional chains (source → condition → target)
        for cond_name, cond in self.joints.items():
            if cond["type"] != "revolute":
                continue
            children = [j for j in self.child_map[cond["child"]]]
            if not children:
                continue
            parent_joint = self.parent_map.get(cond["parent"])
            if parent_joint is None:
                continue
            

            for target in children:
                source = parent_joint
                if source is None:
                    continue
                if "Finger" in source or "Finger" in target or "Thumb" in source or "Thumb" in target or "Hand" in source or "Hand" in target:
                    continue
                cond_vals = np.linspace(cond["lower"], cond["upper"], samples)
                aligned_intervals = []
                current_interval = None

                for val in cond_vals:
                    joint_values = {cond_name: val}
                    src_axis = self.forward_axis(source, joint_values)
                    tgt_axis = self.forward_axis(target, joint_values)
                    dot = np.dot(src_axis, tgt_axis)

                    if abs(dot) >= tol:
                        if current_interval is None:
                            current_interval = [val, val]
                        else:
                            current_interval[1] = val
                    else:
                        if current_interval:
                            aligned_intervals.append(tuple(current_interval))
                            current_interval = None
                if current_interval:
                    aligned_intervals.append(tuple(current_interval))

                for interval in aligned_intervals:
                    relations.append({
                        "target": target,
                        "source": source,
                        "inverse": False,
                        "condition": cond_name,
                        "lower_limit": float(interval[0]),
                        "upper_limit": float(interval[1])
                    })

        # 2. Check direct parent–child pairs (source → target, no condition)
        for joint_name, joint in self.joints.items():
            parent_joint = self.parent_map.get(joint["parent"])
            if parent_joint is None:
                continue
            
            source, target = parent_joint, joint_name
            if "Finger" in source or "Finger" in target or "Thumb" in source or "Thumb" in target or "Hand" in source or "Hand" in target:
                    continue
            joint_values = {}  # no condition
            src_axis = self.forward_axis(source, joint_values)
            tgt_axis = self.forward_axis(target, joint_values)
            dot = np.dot(src_axis, tgt_axis)
            if abs(dot) >= tol:
                relations.append({
                    "target": target,
                    "source": source,
                    "inverse": (dot < 0),
                    "condition": None,
                    "lower_limit": None,
                    "upper_limit": None
                })

        return relations


if __name__ == "__main__":
    
    #urdf_file = "pepper_urdf/pepper_pruned.urdf"  # replace with your URDF path
    #urdf_file = "go2_urdf/go2_description.urdf"  # replace with your URDF path
    urdf_file = "kinova_urdf/gen3_lite.urdf"  # replace with your URDF path
    analyzer = URDFKinematicAnalyzer(urdf_file)
    relations = analyzer.find_relations(samples=200)

    for i, rel in enumerate(relations, 1):
        print(f"relation{i}:")
        for k, v in rel.items():
            print(f"    {k}: {v}")
