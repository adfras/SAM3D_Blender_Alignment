import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import skeleton_core

def main():
    print("Generating Superimposed Visualization...")
    
    # Load and Process Data
    motion_data, blender_rest, hierarchy = skeleton_core.load_data()
    joints_info = skeleton_core.get_blender_joints(blender_rest)
    
    # 1. Align Blender to T-Pose
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_clavicle', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_clavicle', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_uparm', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_lowarm', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_wrist', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_uparm', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_lowarm', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_wrist', [1, 0, 0])

    # 2. Process SAM3D
    sam_joints_raw, sam_bones_indices, sam_names = skeleton_core.get_sam_joints(motion_data, hierarchy, frame_idx=0)
    sam_joints_aligned, sam_height = skeleton_core.initial_sam_alignment(sam_joints_raw)
    bl_height = skeleton_core.compute_blender_height(joints_info)
    
    # 3. Scale
    scale = bl_height / sam_height if sam_height > 1e-6 else 1.0
    sam_joints_aligned *= scale
    
    # 4. Full Reconstruction
    skeleton_core.align_full_skeleton(sam_joints_aligned, joints_info, hierarchy, sam_names)

    # --- Plotting ---
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Superimposed Alignment (Green=Blender, Red=SAM3D)", color='white')

    # Plot Blender (Green)
    bl_parents = hierarchy["parents"]
    bl_names = hierarchy["joints"]
    for i, p_idx in enumerate(bl_parents):
        if p_idx == -1:
            continue
        child_name = bl_names[i]
        parent_name = bl_names[p_idx]
        
        # Skip twist bones for cleaner viz
        is_twist = "twist" in child_name or "proc" in child_name or "null" in child_name
        if is_twist:
            continue
            
        if child_name in joints_info and parent_name in joints_info:
            h_child = skeleton_core.get_joint_point(joints_info[child_name], child_name)
            h_parent = skeleton_core.get_joint_point(joints_info[parent_name], parent_name)
            p1 = skeleton_core.to_plot_coords(h_child)
            p2 = skeleton_core.to_plot_coords(h_parent)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#00ff00', linewidth=3, alpha=0.5)

    # Manually draw forearm->wrist if skipped due to twist parents
    for side in ['l', 'r']:
        elbow = f"{side}_lowarm"
        wrist = f"{side}_wrist"
        if elbow in joints_info and wrist in joints_info:
            h_child = skeleton_core.get_joint_point(joints_info[wrist], wrist)
            h_parent = skeleton_core.get_joint_point(joints_info[elbow], elbow)
            p1 = skeleton_core.to_plot_coords(h_child)
            p2 = skeleton_core.to_plot_coords(h_parent)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#00ff00', linewidth=3, alpha=0.5)

    # Plot SAM3D (Red)
    for p1_idx, p2_idx in sam_bones_indices:
        if p1_idx < len(sam_joints_aligned) and p2_idx < len(sam_joints_aligned):
            p1 = sam_joints_aligned[p1_idx]
            p2 = sam_joints_aligned[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#ff0000', linewidth=1.5, alpha=1.0)

    # Dynamic Zoom/Limits
    min_x, min_y, min_z = np.min(sam_joints_aligned, axis=0)
    max_x, max_y, max_z = np.max(sam_joints_aligned, axis=0)
    pad = 10
    ax.set_xlim3d([min_x - pad, max_x + pad])
    ax.set_ylim3d([min_y - pad, max_y + pad])
    ax.set_zlim3d([min_z - pad, max_z + pad])

    ax.set_xlabel('X')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Up)')
    ax.view_init(elev=10, azim=-90)

    output_file = 'comparison_superimposed.png'
    print(f"Saving {output_file}...")
    plt.savefig(output_file)
    print("Done.")

if __name__ == "__main__":
    main()
