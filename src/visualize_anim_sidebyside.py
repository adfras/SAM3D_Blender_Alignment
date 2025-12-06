import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import skeleton_core
import os

# --- Configuration ---
MOTION_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "idle_motion.json")

def load_motion_data():
    with open(MOTION_FILE, 'r') as f:
        return json.load(f)

def main():
    print("Starting Animation Visualization...")
    
    if not os.path.exists(MOTION_FILE):
        print(f"Error: Motion file not found at {MOTION_FILE}")
        print("Please run 'src/convert_fbx_to_json.py' in Blender first.")
        return

    frames = load_motion_data()
    _, _, hierarchy = skeleton_core.load_data()
    
    # Setup Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare Base SAM Skeleton (for structure/indices)
    # We need a valid SAM skeleton structure to pass to align_full_skeleton
    sam_joints_raw, _, sam_names = skeleton_core.get_sam_joints(skeleton_core.load_data()[0], hierarchy, frame_idx=0)
    sam_joints_base, sam_height = skeleton_core.initial_sam_alignment(sam_joints_raw)
    
    # Helper to get plot coords from Blender coords
    def get_coords(joints_dict, name, point_type='head'):
        if name not in joints_dict:
            return np.array([0,0,0])
        
        # Handle new format (dict with head/tail) or old format (list)
        data = joints_dict[name]
        if isinstance(data, dict):
            raw = np.array(data[point_type])
        else:
            raw = np.array(data)
            
        # Blender (x, y, z) -> Plot (x, y, z)
        # Keep Z as up for Matplotlib 3D
        # Scale by 100 (Meters -> cm)
        return np.array([raw[0], raw[1], raw[2]]) * 100

    # Lines containers
    lines_bl = []
    lines_sam = []
    
    parents = hierarchy["parents"]
    names = hierarchy["joints"]
    
    # Initialize Lines
    for i, p_idx in enumerate(parents):
        if p_idx == -1: continue
        
        name = names[i]
        p_name = names[p_idx]
        
        # Skip twist bones for cleaner viz
        if "twist" in name or "proc" in name: continue
        
        line_bl, = ax.plot([], [], [], color='#00ff00', linewidth=2)
        line_sam, = ax.plot([], [], [], color='#ff0000', linewidth=2)
        
        lines_bl.append((line_bl, name, p_name))
        lines_sam.append((line_sam, i, p_idx)) # SAM uses indices
        
    ax.set_xlim3d([-100, 100])
    ax.set_ylim3d([-100, 100])
    ax.set_zlim3d([0, 200]) # Z is up, character is ~170cm
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Idle Animation: Blender (Green) vs SAM3D Reconstruction (Red)")
    
    # Set Camera View
    ax.view_init(elev=10, azim=-90) # Front view (looking along Y axis)
    
    def update(frame_idx):
        frame_data = frames[frame_idx]["joints"]
        
        # 1. Construct joints_info for this frame (Blender Format)
        current_joints_info = {}
        for name, data in frame_data.items():
            # Data is now {'head': [x,y,z], 'tail': [x,y,z]}
            # We need to scale it for align_full_skeleton
            
            head = np.array(data['head']) * 100
            tail = np.array(data['tail']) * 100
            
            current_joints_info[name] = {'head': head, 'tail': tail}
            
        # 2. Reconstruct SAM Skeleton
        # We start with a copy of the base structure
        current_sam = sam_joints_base.copy()
        
        # Run reconstruction
        # This modifies current_sam in place to match current_joints_info
        skeleton_core.align_full_skeleton(current_sam, current_joints_info, hierarchy, sam_names)
        
        # 3. Update Plots
        
        # Blender (Green) - Offset Left
        for line, name, p_name in lines_bl:
            if name in frame_data and p_name in frame_data:
                c = get_coords(frame_data, name, 'head')
                p = get_coords(frame_data, p_name, 'head')
                
                c[0] -= 50
                p[0] -= 50
                
                line.set_data([c[0], p[0]], [c[1], p[1]])
                line.set_3d_properties([c[2], p[2]])
                
        # SAM3D (Red) - Offset Right
        for line, c_idx, p_idx in lines_sam:
            if c_idx < len(current_sam) and p_idx < len(current_sam):
                c = current_sam[c_idx].copy()
                p = current_sam[p_idx].copy()
                
                c[0] += 50
                p[0] += 50
                
                line.set_data([c[0], p[0]], [c[1], p[1]])
                line.set_3d_properties([c[2], p[2]])
                
        return [l[0] for l in lines_bl] + [l[0] for l in lines_sam]

    print(f"Animating {len(frames)} frames...")
    anim = FuncAnimation(fig, update, frames=len(frames), interval=33, blit=False)
    
    print("Saving animation to animation_sidebyside.gif (this may take a while)...")
    try:
        anim.save('animation_sidebyside.gif', writer='pillow', fps=30)
        print("Saved animation_sidebyside.gif")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        print("Showing plot instead...")
        plt.show()

if __name__ == "__main__":
    main()
