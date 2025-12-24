"""
SAM3D Skeleton for Blender - Single Mesh with Shape Keys
Uses shape keys to animate the skeleton - most reliable method.

HOW TO USE:
1. Open Blender -> Scripting tab
2. Open this file
3. Press Alt+P
"""

import bpy
import json
import os
import numpy as np

# --- Configuration ---
BASE_PATH = r"D:\MediaPipeSAM3D\skeleton_alignment_work\data"
MOTION_PATH = os.path.join(BASE_PATH, "video_motion_full.json")

# MHR70 skeleton connections (same as matplotlib)
MHR70_BONES = [
    (13, 11), (11, 9), (14, 12), (12, 10), (9, 10),  # legs + pelvis
    (5, 9), (6, 10), (5, 6),  # torso
    (5, 69), (6, 69), (69, 0),  # neck/head
    (5, 7), (6, 8), (7, 62), (8, 41),  # arms
]


def align_joints(joints):
    """Same transform as matplotlib."""
    joints = np.array(joints)
    root = (joints[9] + joints[10]) / 2.0
    centered = joints - root

    rotated = np.zeros_like(centered)
    rotated[:, 0] = centered[:, 0]
    rotated[:, 1] = -centered[:, 2]
    rotated[:, 2] = -centered[:, 1]

    min_z = np.min(rotated[:, 2])
    rotated[:, 2] -= min_z
    return rotated


def main():
    print("\n" + "="*50)
    print("SAM3D Skeleton - Shape Key Animation")
    print("="*50)

    with open(MOTION_PATH, 'r') as f:
        motion = json.load(f)

    frames = motion["frames"]
    num_frames = len(frames)
    num_joints = len(frames[0]["joints3d"])
    print(f"Frames: {num_frames}, Joints: {num_joints}")

    # Get all joint indices we need
    needed = set()
    for a, b in MHR70_BONES:
        needed.add(a)
        needed.add(b)
    needed = sorted([i for i in needed if i < num_joints])

    # Map original index to vertex index
    idx_to_vert = {orig: i for i, orig in enumerate(needed)}

    # Pre-align all frames
    print("Aligning all frames...")
    all_aligned = [align_joints(f["joints3d"]) for f in frames]

    # Delete old SAM3D objects
    for obj in list(bpy.data.objects):
        if obj.name.startswith("SAM3D"):
            bpy.data.objects.remove(obj, do_unlink=True)

    # Build mesh vertices and edges from first frame
    first = all_aligned[0]
    verts = [tuple(first[i]) for i in needed]

    edges = []
    for a, b in MHR70_BONES:
        if a in idx_to_vert and b in idx_to_vert:
            edges.append((idx_to_vert[a], idx_to_vert[b]))

    print(f"Creating mesh with {len(verts)} vertices, {len(edges)} edges")

    # Create mesh
    mesh = bpy.data.meshes.new("SAM3D_Mesh")
    mesh.from_pydata(verts, edges, [])
    mesh.update()

    obj = bpy.data.objects.new("SAM3D_Skeleton", mesh)
    bpy.context.collection.objects.link(obj)

    # Red material
    mat = bpy.data.materials.new("SAM3D_Red")
    mat.diffuse_color = (1.0, 0.0, 0.0, 1.0)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)
        bsdf.inputs["Emission Color"].default_value = (1, 0, 0, 1)
        bsdf.inputs["Emission Strength"].default_value = 1.0
    obj.data.materials.append(mat)

    # Skin modifier for thickness
    skin = obj.modifiers.new("Skin", 'SKIN')
    for v in obj.data.skin_vertices[0].data:
        v.radius = (0.025, 0.025)

    # Add basis shape key
    obj.shape_key_add(name="Basis", from_mix=False)

    # Add shape key for each frame and keyframe it
    print("Creating shape keys for animation...")

    for frame_idx in range(num_frames):
        joints = all_aligned[frame_idx]

        # Create shape key
        sk = obj.shape_key_add(name=f"frame_{frame_idx}", from_mix=False)

        # Set vertex positions
        for orig_idx, vert_idx in idx_to_vert.items():
            sk.data[vert_idx].co = tuple(joints[orig_idx])

        # Keyframe: off before, on at frame, off after
        sk.value = 0.0
        if frame_idx > 0:
            sk.keyframe_insert(data_path="value", frame=frame_idx - 1)

        sk.value = 1.0
        sk.keyframe_insert(data_path="value", frame=frame_idx)

        sk.value = 0.0
        if frame_idx < num_frames - 1:
            sk.keyframe_insert(data_path="value", frame=frame_idx + 1)

        if frame_idx % 100 == 0:
            print(f"  Frame {frame_idx}/{num_frames}")

    # Set all shape key interpolation to constant for sharp transitions
    if obj.data.shape_keys and obj.data.shape_keys.animation_data:
        for fc in obj.data.shape_keys.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'

    # Timeline
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1
    bpy.context.scene.frame_set(0)

    # Select object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    obj.show_in_front = True

    print("\n" + "="*50)
    print("DONE! Press SPACE to play animation.")
    print("="*50)


if __name__ == "__main__":
    main()
