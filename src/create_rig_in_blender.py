"""
Create SAM3D Skeleton Rig in Blender and Import Animation

HOW TO USE:
1. Open Blender (fresh scene or your project)
2. Go to Scripting tab
3. Click "Open" and select this file
4. Press Alt+P or click Run Script

This will create the armature and apply animation from your video.
"""

import bpy
import json
import os
import mathutils

# --- Configuration ---
BASE_PATH = r"D:\MediaPipeSAM3D\skeleton_alignment_work\data"
REST_POSE_PATH = os.path.join(BASE_PATH, "blender_rest_pose.json")
HIERARCHY_PATH = os.path.join(BASE_PATH, "mhr_hierarchy.json")
ANIMATION_PATH = os.path.join(BASE_PATH, "animation_data.json")

# Scale factor (rest pose is in cm, convert to meters for Blender)
SCALE = 0.01


def create_armature():
    """Create the SAM3D skeleton as a Blender armature."""
    print("Loading rest pose data...")

    with open(REST_POSE_PATH, 'r') as f:
        rest_pose = json.load(f)

    with open(HIERARCHY_PATH, 'r') as f:
        hierarchy = json.load(f)

    joints = hierarchy["joints"]
    parents = hierarchy["parents"]

    # Create armature
    bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
    armature = bpy.context.object
    armature.name = "SAM3D_Rig"
    arm_data = armature.data
    arm_data.name = "SAM3D_Armature"

    # Remove default bone
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()

    # Create bones
    print(f"Creating {len(joints)} bones...")
    bone_map = {}

    for i, joint_name in enumerate(joints):
        if joint_name not in rest_pose:
            print(f"  Skipping {joint_name} - not in rest pose")
            continue

        bone_data = rest_pose[joint_name]
        head = bone_data["head"]
        tail = bone_data["tail"]

        # Create bone
        bone = arm_data.edit_bones.new(joint_name)

        # Data is already in Blender coordinates (Z-up), just scale to meters
        bone.head = (head[0] * SCALE, head[1] * SCALE, head[2] * SCALE)
        bone.tail = (tail[0] * SCALE, tail[1] * SCALE, tail[2] * SCALE)

        # Ensure bone has some length
        if bone.length < 0.001:
            bone.tail = (bone.head[0], bone.head[1], bone.head[2] + 0.01)

        bone_map[joint_name] = bone

    # Set parents
    print("Setting bone parents...")
    for i, joint_name in enumerate(joints):
        if joint_name not in bone_map:
            continue

        parent_idx = parents[i]
        if parent_idx >= 0:
            parent_name = joints[parent_idx]
            if parent_name in bone_map:
                bone_map[joint_name].parent = bone_map[parent_name]

    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"Armature created with {len(bone_map)} bones")
    return armature


def import_animation(armature):
    """Import animation data onto the armature."""

    if not os.path.exists(ANIMATION_PATH):
        print(f"No animation file found at {ANIMATION_PATH}")
        print("Run retarget_motion.py first to generate animation data.")
        return

    print("Loading animation data...")
    with open(ANIMATION_PATH, 'r') as f:
        anim_data = json.load(f)

    print(f"Importing {len(anim_data)} frames...")

    # Select armature and enter pose mode
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='POSE')

    for frame_entry in anim_data:
        frame_idx = frame_entry["frame"]
        bpy.context.scene.frame_set(frame_idx)

        # Root translation
        root_trans = frame_entry["root_translation"]
        if "root" in armature.pose.bones:
            x, y, z = root_trans
            # Data is already in correct coordinates, just scale
            bl_trans = mathutils.Vector((x * SCALE, y * SCALE, z * SCALE))
            armature.pose.bones["root"].location = bl_trans
            armature.pose.bones["root"].keyframe_insert(data_path="location", frame=frame_idx)

        # Rotations
        rotations = frame_entry["rotations"]
        for bone_name, rot_quat in rotations.items():
            if bone_name not in armature.pose.bones:
                continue

            pb = armature.pose.bones[bone_name]

            # Rotation is [x, y, z, w] from scipy, Blender uses [w, x, y, z]
            rx, ry, rz, rw = rot_quat
            quat = mathutils.Quaternion((rw, rx, ry, rz))

            pb.rotation_mode = 'QUATERNION'
            pb.rotation_quaternion = quat
            pb.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

    bpy.ops.object.mode_set(mode='OBJECT')

    # Set timeline range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(anim_data) - 1
    bpy.context.scene.frame_set(0)

    print("Animation import complete!")


def main():
    print("\n" + "="*50)
    print("SAM3D Rig Creator")
    print("="*50 + "\n")

    # Check files exist
    if not os.path.exists(REST_POSE_PATH):
        print(f"ERROR: Rest pose not found: {REST_POSE_PATH}")
        return

    if not os.path.exists(HIERARCHY_PATH):
        print(f"ERROR: Hierarchy not found: {HIERARCHY_PATH}")
        return

    # Create the rig
    armature = create_armature()

    # Import animation if available
    import_animation(armature)

    print("\n" + "="*50)
    print("DONE! Your SAM3D rig is ready.")
    print("Press SPACE to play animation.")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
