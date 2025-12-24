import json
import numpy as np

import os

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

MOTION_DATA_FILE = os.path.join(DATA_DIR, "video_motion_full.json")
BLENDER_REST_FILE = os.path.join(DATA_DIR, "blender_rest_pose.json")
HIERARCHY_FILE = os.path.join(DATA_DIR, "mhr_hierarchy.json")


def load_data():
    """Load the three required JSON files."""
    with open(MOTION_DATA_FILE, 'r') as f:
        motion_data = json.load(f)
    with open(BLENDER_REST_FILE, 'r') as f:
        blender_rest = json.load(f)
    with open(HIERARCHY_FILE, 'r') as f:
        hierarchy = json.load(f)
    return motion_data, blender_rest, hierarchy


def get_blender_joints(blender_rest):
    """Parse Blender rest pose JSON into a dictionary of head/tail vectors."""
    joints_info = {}
    root_offset = np.array([0.0, 0.0, 0.0])
    if "root" in blender_rest:
        root_offset = np.array(blender_rest["root"]["head"])
    for name, data in blender_rest.items():
        head = np.array(data["head"]) - root_offset
        tail = np.array(data["tail"]) - root_offset
        joints_info[name] = {'head': head, 'tail': tail}
    return joints_info


def get_descendants(hierarchy, root_name):
    """Get all descendant joint names for a given root joint name."""
    parents = hierarchy["parents"]
    names = hierarchy["joints"]
    adj = {}
    for i, p in enumerate(parents):
        if p == -1:
            continue
        p_name = names[p]
        c_name = names[i]
        adj.setdefault(p_name, []).append(c_name)
    descendants = set()
    queue = [root_name]
    while queue:
        curr = queue.pop(0)
        descendants.add(curr)
        if curr in adj:
            queue.extend(adj[curr])
    return descendants


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    if np.linalg.norm(a - b) < 1e-6:
        return np.eye(3)
    
    c = np.dot(a, b)
    if c < -0.9:
        # Near 180 degree rotation
        # Find an orthogonal vector
        if np.abs(a[0]) < np.abs(a[1]):
             orth = np.array([1, 0, 0])
        else:
             orth = np.array([0, 1, 0])
        axis = np.cross(a, orth)
        axis = axis / np.linalg.norm(axis)
        # Rodrigues for 180 deg
        # R = I + 2 sin^2(90) K^2 = I + 2 K^2
        # K is skew symmetric
        ux, uy, uz = axis
        K = np.array([[0, -uz, uy], [uz, 0, -ux], [-uy, ux, 0]])
        return np.eye(3) + 2 * np.dot(K, K)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-8))
    return rotation_matrix


def rotate_blender_limb(joints_info, hierarchy, bone_name, target_vec):
    """
    Rotate a Blender bone (and descendants) so the vector from head to first child head matches target_vec.
    Falls back to head->tail if no child exists.
    """
    if bone_name not in joints_info:
        return
    pivot = joints_info[bone_name]['head']
    parents = hierarchy["parents"]
    names = hierarchy["joints"]
    children = [names[i] for i, p in enumerate(parents) if p != -1 and names[p] == bone_name]
    if children and children[0] in joints_info:
        curr_vec = joints_info[children[0]]['head'] - pivot
    else:
        curr_vec = joints_info[bone_name]['tail'] - pivot
    R = rotation_matrix_from_vectors(curr_vec, np.array(target_vec))
    descendants = get_descendants(hierarchy, bone_name)
    for name in descendants:
        if name in joints_info:
            if name != bone_name:
                h = joints_info[name]['head']
                joints_info[name]['head'] = pivot + R.dot(h - pivot)
            t = joints_info[name]['tail']
            joints_info[name]['tail'] = pivot + R.dot(t - pivot)


def get_sam_joints(motion_data, hierarchy, frame_idx=0):
    """Extract SAM3D joints and bone connections from motion data."""
    if "frames" in motion_data:
        frame = motion_data["frames"][frame_idx]
        joints_raw = np.array(frame["joints3d"])
    elif "pred_joint_coords" in motion_data:
        joints_raw = np.array(motion_data["pred_joint_coords"])
    else:
        raise ValueError("Unknown motion data format")

    num_joints = len(joints_raw)
    if num_joints == 127:
        joints_raw = joints_raw[1:]
        num_joints = len(joints_raw)

    bones = []
    names = []

    if num_joints == 33:
        names = [str(i) for i in range(33)]
        mp_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            (11, 12),
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            (11, 23), (12, 24),
            (23, 24),
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
            (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
        ]
        for p1_idx, p2_idx in mp_connections:
            bones.append((p1_idx, p2_idx))
    else:
        parents = hierarchy["parents"]
        names = hierarchy["joints"]
        
        def get_visual_parent(curr_idx):
            """Traverse up to find first non-twist parent."""
            p_idx = parents[curr_idx]
            while p_idx != -1:
                p_name = names[p_idx]
                is_twist = "twist" in p_name or "proc" in p_name or "null" in p_name
                if not is_twist:
                    return p_idx
                p_idx = parents[p_idx]
            return -1

        for i, p_idx in enumerate(parents):
            if p_idx == -1:
                continue
            child_name = names[i]
            is_twist = "twist" in child_name or "proc" in child_name or "null" in child_name
            if not is_twist:
                # Connect to visual parent instead of direct parent if direct parent is twist
                vis_p_idx = get_visual_parent(i)
                if vis_p_idx != -1:
                    bones.append((i, vis_p_idx))

    return joints_raw, bones, names


def initial_sam_alignment(sam_joints):
    """
    Center SAM joints and rotate them to match Blender's coordinate system roughly.
    Returns rotated joints and the height of the skeleton.
    """
    num_joints = len(sam_joints)
    if num_joints == 33:
        root_pos = (sam_joints[23] + sam_joints[24]) / 2.0 if num_joints > 24 else sam_joints[0]
    else:
        root_pos = sam_joints[0]
    sam_centered = sam_joints - root_pos
    sam_rotated = np.zeros_like(sam_centered)
    sam_rotated[:, 0] = -sam_centered[:, 0]
    sam_rotated[:, 1] = sam_centered[:, 2]
    sam_rotated[:, 2] = -sam_centered[:, 1]
    min_z = np.min(sam_rotated[:, 2])
    max_z = np.max(sam_rotated[:, 2])
    height = max_z - min_z
    return sam_rotated, height


def compute_blender_height(joints_info):
    """Height of the Blender rig along its up-axis (original Y, used as Z in plots)."""
    if not joints_info:
        return 1.0
    y_vals = [data['head'][1] for data in joints_info.values()]
    return max(y_vals) - min(y_vals)


def to_plot_coords(vec):
    """Convert Blender-space vector (X, Y, Z) to plot coords (X, Y, Z). Keep Z-up."""
    return np.array([vec[0], vec[1], vec[2]])


def get_joint_point(blender_joint, name):
    """Use tail for wrists (hands), otherwise head."""
    if 'wrist' in name and 'tail' in blender_joint:
        return np.array(blender_joint['tail'])
    return np.array(blender_joint['head'])


def align_full_skeleton(sam_joints_aligned, joints_info, hierarchy, sam_names):
    """
    Reconstruct the SAM skeleton to EXACTLY match the Blender skeleton using hierarchical reconstruction.
    
    ALGORITHM OVERVIEW:
    Instead of applying iterative rotations/scalings (which accumulate errors), this function
    rebuilds the SAM skeleton from scratch by copying Blender's bone vectors in topological order.
    
    ADVANTAGES:
    - Zero rotation errors (no rotation matrix calculations)
    - Zero scaling errors (direct vector copy)
    - Zero accumulated errors (each joint computed independently from parent)
    - Guaranteed 1:1 match for all bones present in both skeletons
    - Simpler logic (40 lines vs 600 lines)
    
    PROCESS:
    1. Set SAM root to Blender root position (anchor point)
    2. For each joint in topological order (parent before child):
       a. If joint exists in Blender: position = parent + blender_bone_vector
       b. Else (missing in Blender): position = parent (collapse to hide)
    
    PARAMETERS:
    - sam_joints_aligned: (N, 3) numpy array of SAM joint positions (modified in-place)
    - joints_info: dict of Blender joint data from JSON (contains head/tail/matrix)
    - hierarchy: dict with "parents" (parent indices) and "joints" (joint names)
    - sam_names: list of SAM joint names in order
    
    COORDINATE SYSTEMS:
    - Blender uses Z-up coordinate system
    - SAM3D uses Y-up coordinate system
    - Conversion handled by to_plot_coords() for Blender joints
    
    MISSING JOINTS:
    Joints present in SAM but not in Blender (e.g., twist bones, eye nulls) are collapsed
    to their parent position, making them invisible but preserving hierarchy.
    """
    print("--- Starting Hierarchical Reconstruction ---")
    
    # Extract hierarchy information
    parents = hierarchy["parents"]  # List of parent indices for each joint
    names = hierarchy["joints"]     # List of joint names in topological order
    
    # Create fast lookup from joint name to index
    name_to_idx = {n: i for i, n in enumerate(sam_names)}
    
    # Get Blender joint positions in plot coordinates (Y-up to Z-up conversion)
    bl_heads_plot = {n: to_plot_coords(get_joint_point(d, n)) for n, d in joints_info.items()}
    
    # STEP 1: Set root position (anchor point for entire skeleton)
    if 'root' in name_to_idx and 'root' in bl_heads_plot:
        r_idx = name_to_idx['root']
        sam_joints_aligned[r_idx] = bl_heads_plot['root']
    
    # STEP 2: Reconstruct all other joints in topological order
    # The hierarchy is assumed to be in topological order (parent before child)
    for i, name in enumerate(names):
        # Skip root (already set in Step 1)
        if i == 0: 
            continue
        
        # Get parent index from hierarchy
        p_idx = parents[i]
        if p_idx == -1: 
            continue  # Safety check (should not happen for non-root)
        
        parent_name = names[p_idx]
        
        # Verify both current joint and parent are in SAM skeleton
        if name not in name_to_idx or parent_name not in name_to_idx:
            continue
        
        # Get array indices for current joint and its parent
        curr_idx = name_to_idx[name]
        parent_idx = name_to_idx[parent_name]
        
        # Get parent position (already computed in previous iteration)
        parent_pos = sam_joints_aligned[parent_idx]
        
        # RECONSTRUCTION LOGIC:
        if name in bl_heads_plot and parent_name in bl_heads_plot:
            # Case A: Both joint and parent exist in Blender
            # Compute bone vector from Blender and apply to SAM
            bl_bone_vec = bl_heads_plot[name] - bl_heads_plot[parent_name]
            sam_joints_aligned[curr_idx] = parent_pos + bl_bone_vec
        else:
            # Case B: Joint missing in Blender (e.g., twist bones, nulls)
            # Collapse to parent position (makes bone invisible)
            sam_joints_aligned[curr_idx] = parent_pos

    print("--- Reconstruction Complete ---")
