import json
import numpy as np
import skeleton_core
from scipy.spatial.transform import Rotation as R

def get_bone_vector(joints, p_idx, c_idx):
    return joints[c_idx] - joints[p_idx]

def calculate_local_rotations(sam_joints, hierarchy, sam_names, rest_joints_info):
    """
    Calculate local rotations (quaternions) for each bone to match the SAM joints.
    Assumes sam_joints are already aligned/scaled to match Blender size.
    """
    parents = hierarchy["parents"]
    names = hierarchy["joints"]
    name_to_idx = {n: i for i, n in enumerate(sam_names)}
    
    # Global rotations for each bone (relative to world)
    global_rotations = {} # name -> Rotation object
    
    # Local rotations (relative to parent)
    local_rotations = {} # name -> [x, y, z, w]
    
    # Root Translation
    root_name = "root"
    root_translation = [0, 0, 0]
    if root_name in name_to_idx:
        root_translation = sam_joints[name_to_idx[root_name]].tolist()
        # Initialize root global rotation as identity (will be updated if root has rotation)
        global_rotations[root_name] = R.identity()

    # Traverse hierarchy
    for i, name in enumerate(names):
        if i == 0: continue # Skip root for rotation logic (handled separately or assumed identity)
        
        p_idx_hier = parents[i]
        if p_idx_hier == -1: continue
        
        parent_name = names[p_idx_hier]
        
        # Check if we have data for this bone
        if name not in name_to_idx or parent_name not in name_to_idx:
            # If missing, assume identity relative to parent
            global_rotations[name] = global_rotations.get(parent_name, R.identity())
            local_rotations[name] = [0, 0, 0, 1]
            continue

        # Get indices in SAM array
        curr_idx = name_to_idx[name]
        parent_idx = name_to_idx[parent_name]

        # Check bounds - SAM3D may output fewer joints than hierarchy defines
        num_joints = len(sam_joints)
        if curr_idx >= num_joints or parent_idx >= num_joints:
            global_rotations[name] = global_rotations.get(parent_name, R.identity())
            local_rotations[name] = [0, 0, 0, 1]
            continue

        # 1. Get Target Vector (Current Pose)
        target_vec = sam_joints[curr_idx] - sam_joints[parent_idx]
        if np.linalg.norm(target_vec) < 1e-6:
            global_rotations[name] = global_rotations.get(parent_name, R.identity())
            local_rotations[name] = [0, 0, 0, 1]
            continue
            
        # 2. Get Rest Vector (Blender Rest Pose)
        # We need the vector from parent HEAD to child HEAD (or tail if leaf)
        # skeleton_core.get_joint_point handles the head/tail logic
        if name in rest_joints_info and parent_name in rest_joints_info:
             # Note: get_joint_point returns Blender coords (X, Y, Z). 
             # We need to convert to the same space as SAM joints.
             # SAM joints in this script are expected to be in "Plot Coords" (X, -Z, Y) 
             # because that's what skeleton_core.align_full_skeleton produces/uses.
             
             p_rest_bl = skeleton_core.get_joint_point(rest_joints_info[parent_name], parent_name)
             c_rest_bl = skeleton_core.get_joint_point(rest_joints_info[name], name)
             
             p_rest = skeleton_core.to_plot_coords(p_rest_bl)
             c_rest = skeleton_core.to_plot_coords(c_rest_bl)
             
             rest_vec = c_rest - p_rest
        else:
            # Fallback
            rest_vec = target_vec # No rotation

        # 3. Calculate Rotation to align Rest to Target
        # This gives us the rotation from Rest Orientation to Target Orientation
        # BUT: This is only the "Swing". Twist is unknown.
        # We assume Minimal Twist (swing-only).
        
        # rotation_matrix_from_vectors returns a 3x3 matrix
        rot_mat = skeleton_core.rotation_matrix_from_vectors(rest_vec, target_vec)
        r_swing = R.from_matrix(rot_mat)
        
        # Global Rotation of this bone = Swing * (Parent Global Rotation?)
        # Actually, the "Rest Vector" is defined in World Space (T-Pose).
        # So 'r_swing' IS the rotation from the T-Pose vector to the Current Vector.
        # If the bone in T-Pose was aligned with World Axes, r_swing would be the absolute orientation.
        # But it's not.
        
        # Let's define:
        # R_global = R_swing * R_rest_global
        # But we don't track R_rest_global explicitly.
        # Instead, we just want the rotation that DEFORMS the rest pose.
        # Blender Pose Bones apply rotation ON TOP of the Rest Pose.
        # So if we calculate the rotation that takes RestVector -> PoseVector, 
        # that IS the rotation we want to apply to the PoseBone (in Global space).
        
        r_global_deformation = r_swing
        
        # Now convert to Local Space (relative to parent's deformation)
        # R_global_child = R_global_parent * R_local_child
        # R_local_child = inv(R_global_parent) * R_global_child
        
        parent_rot = global_rotations.get(parent_name, R.identity())
        r_local = parent_rot.inv() * r_global_deformation
        
        # Store
        global_rotations[name] = r_global_deformation
        local_rotations[name] = r_local.as_quat().tolist() # x, y, z, w

    return root_translation, local_rotations

def main():
    print("Starting Animation Retargeting...")
    
    # Load Data
    motion_data, blender_rest, hierarchy = skeleton_core.load_data()
    joints_info = skeleton_core.get_blender_joints(blender_rest)
    
    # Prepare frames
    # Check if motion_data is a list (multiple frames) or dict (single frame)
    frames = []
    if isinstance(motion_data, list):
        frames = motion_data
    elif "frames" in motion_data:
        frames = motion_data["frames"]
    elif "pred_keypoints_3d" in motion_data:
        # Single frame wrapped in dict
        frames = [motion_data]
    
    print(f"Found {len(frames)} frames.")
    
    animation_data = []
    
    for f_idx, frame_data in enumerate(frames):
        # 1. Parse SAM Joints for this frame
        # We construct a dummy object if needed to reuse skeleton_core.get_sam_joints
        # skeleton_core.get_sam_joints expects the full motion_data dict structure
        # or a dict with "pred_joint_coords".
        
        # If frame_data is just the list of points, wrap it.
        temp_data = {}
        if "pred_keypoints_3d" in frame_data:
             temp_data["pred_keypoints_3d"] = frame_data["pred_keypoints_3d"]
        elif "joints3d" in frame_data:
             temp_data["pred_joint_coords"] = frame_data["joints3d"] # Map to expected key
        else:
             # Assume frame_data IS the list of points
             temp_data["pred_joint_coords"] = frame_data
             
        sam_joints_raw, _, sam_names = skeleton_core.get_sam_joints(temp_data, hierarchy, frame_idx=0)
        
        # 2. Align & Scale (Standard Pipeline)
        sam_joints_aligned, sam_height = skeleton_core.initial_sam_alignment(sam_joints_raw)
        bl_height = skeleton_core.compute_blender_height(joints_info)
        scale = bl_height / sam_height if sam_height > 1e-6 else 1.0
        sam_joints_aligned *= scale
        
        # 3. Calculate Rotations
        root_trans, local_rots = calculate_local_rotations(sam_joints_aligned, hierarchy, sam_names, joints_info)
        
        animation_data.append({
            "frame": f_idx,
            "root_translation": root_trans,
            "rotations": local_rots
        })
        
        if f_idx % 10 == 0:
            print(f"Processed frame {f_idx}/{len(frames)}")

    # Save
    out_file = "data/animation_data.json"
    with open(out_file, "w") as f:
        json.dump(animation_data, f, indent=2)
    
    print(f"Saved animation data to {out_file}")

if __name__ == "__main__":
    main()
