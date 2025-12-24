"""
Side-by-side skeleton comparison visualization.
Shows Blender (green) and SAM3D (red) skeletons separated for easier visual comparison.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append('.')
import skeleton_core

def main():
    print("Starting Side-by-Side Comparison...")
    
    # Load data
    motion_data, blender_rest, hierarchy = skeleton_core.load_data()
    
    # Process Blender skeleton
    joints_info = skeleton_core.get_blender_joints(blender_rest)
    
    # Align Blender to T-pose
    print("Aligning Blender Skeleton to T-Pose...")
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_clavicle', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_clavicle', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_uparm', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_lowarm', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_wrist', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_uparm', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_lowarm', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_wrist', [1, 0, 0])
    
    # Process SAM3D skeleton
    sam_joints_raw, sam_bones_indices, sam_names = skeleton_core.get_sam_joints(motion_data, hierarchy, frame_idx=0)
    sam_joints_aligned, sam_height = skeleton_core.initial_sam_alignment(sam_joints_raw)
    bl_height = skeleton_core.compute_blender_height(joints_info)
    
    # Scale SAM3D to match Blender height
    scale = bl_height / sam_height if sam_height > 1e-6 else 1.0
    sam_joints_aligned *= scale
    
    # Apply reconstruction
    print("\n--- Starting Hierarchical Reconstruction ---")
    skeleton_core.align_full_skeleton(sam_joints_aligned, joints_info, hierarchy, sam_names)
    
    # OFFSET: Move SAM3D to the right, Blender to the left
    # Calculate skeleton width for spacing
    bl_positions = np.array([skeleton_core.to_plot_coords(skeleton_core.get_joint_point(d, n)) 
                            for n, d in joints_info.items()])
    width = max(np.max(bl_positions[:, 0]), np.max(sam_joints_aligned[:, 0])) - \
            min(np.min(bl_positions[:, 0]), np.min(sam_joints_aligned[:, 0]))
    
    spacing = width * 0.6  # 60% of width as gap
    
    # Offset Blender to the left
    bl_offset = np.array([-spacing, 0, 0])
    
    # Offset SAM3D to the right  
    sam_offset = np.array([spacing, 0, 0])
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#1a1a1a')
    
    # Plot Blender skeleton (GREEN, left side)
    for name, data in joints_info.items():
        parent_idx = hierarchy["parents"][hierarchy["joints"].index(name)]
        if parent_idx == -1:
            continue
        
        parent_name = hierarchy["joints"][parent_idx]
        if parent_name not in joints_info:
            continue
        
        h_child = skeleton_core.get_joint_point(data, name)
        h_parent = skeleton_core.get_joint_point(joints_info[parent_name], parent_name)
        
        # Convert to plot coords and apply offset
        p1 = skeleton_core.to_plot_coords(h_child) + bl_offset
        p2 = skeleton_core.to_plot_coords(h_parent) + bl_offset
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                color='#00ff00', linewidth=2, alpha=0.7, label='Blender' if name == 'root' else '')
    
    # Add label for Blender
    bl_root = skeleton_core.to_plot_coords(joints_info['root']['head']) + bl_offset
    ax.text(bl_root[0], bl_root[1], bl_root[2] + 20, "BLENDER\n(Reference)", 
            color='#00ff00', fontsize=14, ha='center', weight='bold')
    
    # Plot SAM3D skeleton (RED, right side)
    sam_joints_offset = sam_joints_aligned + sam_offset
    
    for p1_idx, p2_idx in sam_bones_indices:
        if p1_idx < len(sam_joints_offset) and p2_idx < len(sam_joints_offset):
            p1 = sam_joints_offset[p1_idx]
            p2 = sam_joints_offset[p2_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                    color='#ff0000', linewidth=2, alpha=0.8, label='SAM3D' if p1_idx == 0 else '')
    
    # Add label for SAM3D
    sam_root = sam_joints_offset[0]
    ax.text(sam_root[0], sam_root[1], sam_root[2] + 20, "SAM3D\n(Reconstructed)", 
            color='#ff0000', fontsize=14, ha='center', weight='bold')
    
    # Set dynamic limits
    all_points = np.vstack([bl_positions + bl_offset, sam_joints_offset])
    min_x, min_y, min_z = np.min(all_points, axis=0)
    max_x, max_y, max_z = np.max(all_points, axis=0)
    
    pad = 20
    ax.set_xlim3d([min_x - pad, max_x + pad])
    ax.set_ylim3d([min_y - pad, max_y + pad])
    ax.set_zlim3d([min_z - pad, max_z + pad])
    
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y (Depth)', color='white')
    ax.set_zlabel('Z (Up)', color='white')
    ax.tick_params(colors='white')
    ax.view_init(elev=10, azim=-90)
    
    # Add title
    plt.title('Side-by-Side Skeleton Comparison\nGreen = Blender (Reference) | Red = SAM3D (Reconstructed)', 
              color='white', fontsize=16, pad=20)
    
    print("Saving Side-by-Side Comparison to comparison_sidebyside.png...")
    plt.savefig('comparison_sidebyside.png', facecolor='#1a1a1a', dpi=150)
    print("✓ Saved comparison_sidebyside.png")
    plt.show()

if __name__ == "__main__":
    main()
