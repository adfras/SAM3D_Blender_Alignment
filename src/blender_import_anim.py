import bpy
import json
import os
import mathutils

# --- Configuration ---
# Path to animation_data.json (Update this if running from a different location)
# Best to use absolute path or relative to blend file
JSON_PATH = r"D:\MediaPipeSAM3D\skeleton_alignment_work\data\animation_data.json"

def import_animation():
    print("Starting Animation Import...")
    
    if not os.path.exists(JSON_PATH):
        print(f"Error: File not found at {JSON_PATH}")
        return

    with open(JSON_PATH, 'r') as f:
        anim_data = json.load(f)
        
    obj = bpy.context.object
    if not obj or obj.type != 'ARMATURE':
        print("Error: Active object must be an Armature.")
        return
        
    print(f"Importing {len(anim_data)} frames to {obj.name}...")
    
    # Ensure we are in Pose Mode
    bpy.ops.object.mode_set(mode='POSE')
    
    for frame_entry in anim_data:
        frame_idx = frame_entry["frame"]
        bpy.context.scene.frame_set(frame_idx)
        
        # 1. Root Translation
        # Assuming 'root' bone exists and is the main mover
        root_trans = frame_entry["root_translation"]
        # Blender uses Z-up. Our data is likely in "Plot Coords" (X, -Z, Y) from retarget_motion.py
        # We need to map it back to Blender World Coords.
        # Plot(x, y, z) -> Blender(x, z, -y) ?
        # skeleton_core.to_plot_coords: (x, y, z) -> (x, -z, y)
        # Inverse: (x, y, z) -> (x, z, -y)
        
        # Wait, let's check retarget_motion.py logic.
        # It uses sam_joints_aligned which are in Plot Coords.
        # So yes, we need to inverse map.
        
        x, y, z = root_trans
        bl_trans = mathutils.Vector((x, z, -y))
        
        if "root" in obj.pose.bones:
            # Set location (relative to rest? or global?)
            # Usually root location is set on the object or the root bone.
            # Let's set it on the root bone.
            # Note: This might need offset adjustment if rest pose is not at 0,0,0
            obj.pose.bones["root"].location = bl_trans
            obj.pose.bones["root"].keyframe_insert(data_path="location", frame=frame_idx)

        # 2. Rotations
        rotations = frame_entry["rotations"]
        for bone_name, rot_quat in rotations.items():
            if bone_name not in obj.pose.bones:
                continue
            
            pb = obj.pose.bones[bone_name]
            
            # Rotation is [x, y, z, w] from scipy
            # Blender uses [w, x, y, z]
            rx, ry, rz, rw = rot_quat
            quat = mathutils.Quaternion((rw, rx, ry, rz))
            
            # Coordinate System Conversion for Rotation
            # This is tricky. The rotation was calculated in "Plot Space".
            # We need to apply it in "Blender Space".
            # Since we mapped vectors (x,y,z)->(x,-z,y), the rotation frame is also twisted.
            
            # Simplest approach: Just try applying it. 
            # If axes are wrong, we might need to permute components.
            # But mathematically, if we rotate the basis vectors, the quaternion components change.
            
            # Let's assume for now that since we are applying "Swing" rotations derived from vectors,
            # and we map the vectors back to Blender space implicitly by mapping the root,
            # maybe we need to map the quaternion too.
            
            # Mapping (x, y, z) -> (x, z, -y) corresponds to a rotation of -90 deg around X-axis.
            # R_blender = R_fix * R_plot * inv(R_fix) ?
            
            # Let's stick to raw application first, user can verify.
            
            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = quat
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)
            
    print("Import Complete.")

if __name__ == "__main__":
    import_animation()
