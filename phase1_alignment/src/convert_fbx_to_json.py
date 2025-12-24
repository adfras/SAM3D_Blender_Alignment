import bpy
import json
import os

# --- Configuration ---
FBX_PATH = r"D:\MediaPipeSAM3D\skeleton_alignment_work\Idle.fbx"
OUTPUT_PATH = r"D:\MediaPipeSAM3D\skeleton_alignment_work\data\idle_motion.json"

def convert_fbx():
    print("Starting FBX Conversion...")
    
    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Import FBX
    if not os.path.exists(FBX_PATH):
        print(f"Error: FBX not found at {FBX_PATH}")
        return
        
    bpy.ops.import_scene.fbx(filepath=FBX_PATH)
    
    # Find Armature
    armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
            
    if not armature:
        print("Error: No armature found in FBX.")
        return
        
    print(f"Processing Armature: {armature.name}")
    
    # Get Frame Range
    scene = bpy.context.scene
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    print(f"Frames: {start_frame} to {end_frame}")
    
    motion_data = []
    
    # Bone Mapping (Mixamo -> MHR)
    bone_map = {
        "mixamorig:Hips": "root",
        "mixamorig:Spine": "c_spine0",
        "mixamorig:Spine1": "c_spine1",
        "mixamorig:Spine2": "c_spine2",
        "mixamorig:Neck": "c_neck",
        "mixamorig:Head": "c_head",
        "mixamorig:LeftShoulder": "l_clavicle",
        "mixamorig:LeftArm": "l_uparm",
        "mixamorig:LeftForeArm": "l_lowarm",
        "mixamorig:LeftHand": "l_wrist",
        "mixamorig:RightShoulder": "r_clavicle",
        "mixamorig:RightArm": "r_uparm",
        "mixamorig:RightForeArm": "r_lowarm",
        "mixamorig:RightHand": "r_wrist",
        "mixamorig:LeftUpLeg": "l_upleg",
        "mixamorig:LeftLeg": "l_lowleg",
        "mixamorig:LeftFoot": "l_foot",
        "mixamorig:RightUpLeg": "r_upleg",
        "mixamorig:RightLeg": "r_lowleg",
        "mixamorig:RightFoot": "r_foot"
    }

    for f in range(start_frame, end_frame + 1):
        scene.frame_set(f)
        
        frame_data = {
            "frame": f,
            "joints": {}
        }
        
        # Record Global Head Positions for each bone
        for bone in armature.pose.bones:
            # Map name
            target_name = bone_map.get(bone.name, bone.name) # Fallback to original if no map
            
            # Global Head and Tail Position
            global_head = armature.matrix_world @ bone.head
            global_tail = armature.matrix_world @ bone.tail
            
            frame_data["joints"][target_name] = {
                "head": [global_head.x, global_head.y, global_head.z],
                "tail": [global_tail.x, global_tail.y, global_tail.z]
            }
            
        motion_data.append(frame_data)
        
    # Save to JSON
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(motion_data, f, indent=2)
        
    print(f"Saved motion data to {OUTPUT_PATH}")

if __name__ == "__main__":
    convert_fbx()
