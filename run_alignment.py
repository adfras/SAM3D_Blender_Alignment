import json
import numpy as np
import csv
import skeleton_core

def main():
    print("Starting Skeleton Alignment...")
    motion_data, blender_rest, hierarchy = skeleton_core.load_data()

    # 1. Parse Data
    joints_info = skeleton_core.get_blender_joints(blender_rest)
    sam_joints_raw, sam_bones_indices, sam_names = skeleton_core.get_sam_joints(motion_data, hierarchy, frame_idx=0)
    
    # 2. Initial Alignment (Rotation & Scale)
    sam_joints_aligned, sam_height = skeleton_core.initial_sam_alignment(sam_joints_raw)
    bl_height = skeleton_core.compute_blender_height(joints_info)

    # 3. Rotate Blender to T-Pose
    print("Aligning Blender Skeleton to T-Pose...")
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_clavicle', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_clavicle', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_uparm', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_lowarm', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_wrist', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_uparm', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_lowarm', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_wrist', [1, 0, 0])

    # 4. Scale SAM to match Blender
    scale = bl_height / sam_height if sam_height > 1e-6 else 1.0
    print(f"Applying Scale: {scale:.2f} (SAM height -> Blender height)")
    sam_joints_aligned *= scale

    # 5. Full Hierarchical Reconstruction (The "Perfect Alignment")
    skeleton_core.align_full_skeleton(sam_joints_aligned, joints_info, hierarchy, sam_names)

    # 6. Save Outputs
    save_joint_positions(joints_info, sam_joints_aligned, sam_names)
    save_bone_vectors(joints_info, sam_joints_aligned, sam_names)
    save_aligned_json(sam_joints_aligned, sam_names)

    print("Alignment Complete.")


def save_joint_positions(joints_info, sam_joints_aligned, sam_names):
    filename = "debug_joint_positions.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Skeleton", "JointName", "X", "Y", "Z"])
        
        # Blender (Head positions as proxy for joints)
        for name, data in joints_info.items():
            # Convert to plot coords to match visualization
            head = skeleton_core.get_joint_point(data, name)
            p = skeleton_core.to_plot_coords(head)
            writer.writerow(["Blender", name, p[0], p[1], p[2]])

        # SAM3D
        for i, name in enumerate(sam_names):
            p = sam_joints_aligned[i]
            writer.writerow(["SAM3D", name, p[0], p[1], p[2]])
    print(f"Saved {filename}")


def save_bone_vectors(joints_info, sam_joints_aligned, sam_names):
    filename = "debug_bone_vectors.csv"
    bone_map = [
        ('Spine', 'root', 'spine1'),
        ('Neck', 'c_neck', 'c_head'),
        ('R_Clavicle', 'r_shoulder', 'r_uparm'),
        ('R_UpperArm', 'r_uparm', 'r_lowarm'),
        ('R_Forearm', 'r_lowarm', 'r_wrist'),
        ('L_Clavicle', 'l_shoulder', 'l_uparm'),
        ('L_UpperArm', 'l_uparm', 'l_lowarm'),
        ('L_Forearm', 'l_lowarm', 'l_wrist'),
        ('R_Thigh', 'r_upleg', 'r_lowleg'),
        ('R_Shin', 'r_lowleg', 'r_foot'),
        ('L_Thigh', 'l_upleg', 'l_lowleg'),
        ('L_Shin', 'l_lowleg', 'l_foot'),
        # Fingers
        ('L_Thumb1', 'l_wrist', 'l_thumb1'), ('L_Thumb2', 'l_thumb1', 'l_thumb2'), ('L_Thumb3', 'l_thumb2', 'l_thumb3'),
        ('L_Index1', 'l_wrist', 'l_index1'), ('L_Index2', 'l_index1', 'l_index2'), ('L_Index3', 'l_index2', 'l_index3'),
        ('L_Middle1', 'l_wrist', 'l_middle1'), ('L_Middle2', 'l_middle1', 'l_middle2'), ('L_Middle3', 'l_middle2', 'l_middle3'),
        ('L_Ring1', 'l_wrist', 'l_ring1'), ('L_Ring2', 'l_ring1', 'l_ring2'), ('L_Ring3', 'l_ring2', 'l_ring3'),
        ('L_Pinky1', 'l_wrist', 'l_pinky1'), ('L_Pinky2', 'l_pinky1', 'l_pinky2'), ('L_Pinky3', 'l_pinky2', 'l_pinky3'),
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Bone", "Blender_Len", "SAM_Len", "Ratio", "Angle_Deg", "Blender_Vec", "SAM_Vec"])

        bl_heads_plot = {n: skeleton_core.to_plot_coords(skeleton_core.get_joint_point(d, n)) for n, d in joints_info.items()}

        for name, start, end in bone_map:
            # Blender
            b_vec = np.zeros(3)
            b_len = 0.0
            if start in bl_heads_plot and end in bl_heads_plot:
                b_vec = bl_heads_plot[end] - bl_heads_plot[start]
                b_len = np.linalg.norm(b_vec)

            # SAM
            s_vec = np.zeros(3)
            s_len = 0.0
            if start in sam_names and end in sam_names:
                s_idx = sam_names.index(start)
                e_idx = sam_names.index(end)
                s_vec = sam_joints_aligned[e_idx] - sam_joints_aligned[s_idx]
                s_len = np.linalg.norm(s_vec)

            # Metrics
            ratio = s_len / b_len if b_len > 1e-6 else 0.0
            
            dot = 0.0
            angle = 0.0
            if b_len > 1e-6 and s_len > 1e-6:
                b_dir = b_vec / b_len
                s_dir = s_vec / s_len
                dot = np.clip(np.dot(b_dir, s_dir), -1.0, 1.0)
                angle = np.degrees(np.arccos(dot))

            writer.writerow([name, f"{b_len:.4f}", f"{s_len:.4f}", f"{ratio:.4f}", f"{angle:.4f}", 
                             f"[{b_vec[0]:.3f}, {b_vec[1]:.3f}, {b_vec[2]:.3f}]",
                             f"[{s_vec[0]:.3f}, {s_vec[1]:.3f}, {s_vec[2]:.3f}]"])
    print(f"Saved {filename}")


def save_aligned_json(sam_joints_aligned, sam_names):
    """Save the aligned SAM skeleton as a JSON file."""
    output_data = {
        "skeleton_type": "SAM3D_Aligned_to_Blender",
        "joints": {}
    }
    for i, name in enumerate(sam_names):
        output_data["joints"][name] = sam_joints_aligned[i].tolist()
    
    with open("aligned_skeleton_data.json", "w") as f:
        json.dump(output_data, f, indent=4)
    print("Saved aligned_skeleton_data.json")


if __name__ == "__main__":
    main()
