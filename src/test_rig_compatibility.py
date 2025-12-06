import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skeleton_core
import math

def main():
    print("Starting Procedural Compatibility Test...")
    
    # Load Data
    motion_data, blender_rest, hierarchy = skeleton_core.load_data()
    joints_info = skeleton_core.get_blender_joints(blender_rest)
    
    # 1. Align Blender to T-Pose (Base State)
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_clavicle', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_clavicle', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_uparm', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_lowarm', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'r_wrist', [-1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_uparm', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_lowarm', [1, 0, 0])
    skeleton_core.rotate_blender_limb(joints_info, hierarchy, 'l_wrist', [1, 0, 0])
    
    # 2. Prepare SAM3D (Base State)
    sam_joints_raw, sam_bones_indices, sam_names = skeleton_core.get_sam_joints(motion_data, hierarchy, frame_idx=0)
    sam_joints_aligned, sam_height = skeleton_core.initial_sam_alignment(sam_joints_raw)
    bl_height = skeleton_core.compute_blender_height(joints_info)
    scale = bl_height / sam_height if sam_height > 1e-6 else 1.0
    sam_joints_aligned *= scale
    
    # Store Base SAM Pose
    skeleton_core.align_full_skeleton(sam_joints_aligned, joints_info, hierarchy, sam_names)
    
    # --- Procedural Animation ---
    # We will animate the Blender skeleton (Green) and show that SAM3D (Red) follows.
    # Animation: Arms waving up and down.
    
    num_frames = 60
    frames = []
    
    import copy
    
    print("Generating frames...")
    for f in range(num_frames):
        # Deep copy joints info to modify it without affecting base
        # joints_info is dict of dicts of numpy arrays. Deep copy needed.
        current_joints = copy.deepcopy(joints_info)
        
        # Calculate wave angle
        angle = math.sin(f / num_frames * 2 * math.pi) * 45 # +/- 45 degrees
        
        # Convert angle to vector direction
        # T-pose is [1, 0, 0] for Left Arm
        # Wave up/down (Z axis)
        rad = math.radians(angle)
        
        # Left Arm (Positive X is out)
        # Up is Positive Z
        lx = math.cos(rad)
        lz = math.sin(rad)
        l_vec = [lx, 0, lz]
        
        # Right Arm (Negative X is out)
        rx = -math.cos(rad)
        rz = math.sin(rad)
        r_vec = [rx, 0, rz]
        
        # Apply Rotation to Blender Skeleton
        skeleton_core.rotate_blender_limb(current_joints, hierarchy, 'l_uparm', l_vec)
        skeleton_core.rotate_blender_limb(current_joints, hierarchy, 'r_uparm', r_vec)
        
        # Reconstruct SAM3D from this new Blender pose
        # We start with the base aligned SAM joints to keep the root/scale
        current_sam = sam_joints_aligned.copy()
        skeleton_core.align_full_skeleton(current_sam, current_joints, hierarchy, sam_names)
        
        frames.append((current_joints, current_sam))

    # --- Visualization ---
    print("Visualizing...")
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Helper
    def get_coords(joints_dict, name):
        if name not in joints_dict: return np.array([0,0,0])
        raw = skeleton_core.get_joint_point(joints_dict[name], name)
        return skeleton_core.to_plot_coords(raw)

    lines_bl = []
    lines_sam = []
    
    parents = hierarchy["parents"]
    names = hierarchy["joints"]
    
    # Init Lines
    for i, p_idx in enumerate(parents):
        if p_idx == -1: continue
        name = names[i]
        p_name = names[p_idx]
        
        # Skip twist
        if "twist" in name or "proc" in name: continue
        
        line_bl, = ax.plot([], [], [], color='#00ff00', linewidth=2)
        line_sam, = ax.plot([], [], [], color='#ff0000', linewidth=2)
        lines_bl.append((line_bl, name, p_name))
        lines_sam.append((line_sam, i, p_idx)) # SAM uses indices

    ax.set_xlim3d([-100, 100])
    ax.set_ylim3d([-100, 100])
    ax.set_zlim3d([-100, 100])
    ax.set_title("Compatibility Test: Procedural Wave\nGreen=Blender (Driver) | Red=SAM3D (Follower)")
    
    def update(frame_idx):
        bl_joints, sam_joints = frames[frame_idx]
        
        # Blender (Offset Left)
        for line, name, p_name in lines_bl:
            if name in bl_joints and p_name in bl_joints:
                c = get_coords(bl_joints, name)
                p = get_coords(bl_joints, p_name)
                c[0] -= 50
                p[0] -= 50
                line.set_data([c[0], p[0]], [c[1], p[1]])
                line.set_3d_properties([c[2], p[2]])
                
        # SAM3D (Offset Right)
        for line, c_idx, p_idx in lines_sam:
            if c_idx < len(sam_joints) and p_idx < len(sam_joints):
                c = sam_joints[c_idx].copy()
                p = sam_joints[p_idx].copy()
                c[0] += 50
                p[0] += 50
                line.set_data([c[0], p[0]], [c[1], p[1]])
                line.set_3d_properties([c[2], p[2]])
                
        return [l[0] for l in lines_bl] + [l[0] for l in lines_sam]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    
    try:
        anim.save('test_compatibility.gif', writer='pillow', fps=20)
        print("Saved test_compatibility.gif")
    except:
        plt.show()

if __name__ == "__main__":
    main()
