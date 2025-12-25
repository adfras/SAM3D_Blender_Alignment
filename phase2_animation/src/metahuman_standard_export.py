"""
MetaHuman Standard Export
=========================

Creates skeleton with MetaHuman-standard bone names for direct UE5 retargeting.
Bones are named like 'pelvis', 'spine_01', 'thigh_l' instead of 'pelvis_to_spine_01'.

Key differences from complete_pipeline_metahuman.py:
- Bones have single names matching MetaHuman skeleton
- Proper parent-child hierarchy (not just stretch constraints)
- Better compatibility with UE5 IK Retargeter auto-mapping

Usage in Blender:
    1. Open script in Scripting tab
    2. Press Alt+P to run
    3. Press Spacebar to play animation
"""

import bpy
import json
import os
from mathutils import Matrix, Vector

# --- Configuration ---
try:
    script_path = bpy.context.space_data.text.filepath if bpy.context.space_data and bpy.context.space_data.text else None
    if script_path:
        BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))
    else:
        BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except:
    BASE_PATH = r"D:\MediaPipeSAM3D\skeleton_alignment_work\SAM3D2Blender"

HIERARCHY_PATH = os.path.join(BASE_PATH, "data", "mhr_hierarchy.json")
MOTION_PATH_SMOOTH = os.path.join(BASE_PATH, "data", "video_motion_armature_smooth.json")
MOTION_PATH_RAW = os.path.join(BASE_PATH, "data", "video_motion_armature.json")
MOTION_PATH = MOTION_PATH_SMOOTH if os.path.exists(MOTION_PATH_SMOOTH) else MOTION_PATH_RAW
FBX_OUTPUT_PATH = os.path.join(BASE_PATH, "data", "metahuman_standard.fbx")

COORD_TRANSFORM = Matrix([[1,0,0], [0,0,1], [0,-1,0]])

# SAM3D joint name to MetaHuman bone name mapping
SAM3D_TO_MH = {
    "root": "pelvis",
    "c_spine0": "spine_01",
    "c_spine1": "spine_02",
    "c_spine2": "spine_03",
    "c_spine3": "spine_05",
    "c_neck": "neck_01",
    "c_head": "head",
    "l_clavicle": "clavicle_l",
    "l_uparm": "upperarm_l",
    "l_lowarm": "lowerarm_l",
    "l_wrist": "hand_l",
    "r_clavicle": "clavicle_r",
    "r_uparm": "upperarm_r",
    "r_lowarm": "lowerarm_r",
    "r_wrist": "hand_r",
    "l_upleg": "thigh_l",
    "l_lowleg": "calf_l",
    "l_foot": "foot_l",
    "l_ball": "ball_l",
    "r_upleg": "thigh_r",
    "r_lowleg": "calf_r",
    "r_foot": "foot_r",
    "r_ball": "ball_r",
    # Left fingers (SAM3D has pinky0 as metacarpal, others start at 1)
    "l_thumb0": "thumb_01_l",
    "l_thumb1": "thumb_02_l",
    "l_thumb2": "thumb_03_l",
    "l_index1": "index_01_l",
    "l_index2": "index_02_l",
    "l_index3": "index_03_l",
    "l_middle1": "middle_01_l",
    "l_middle2": "middle_02_l",
    "l_middle3": "middle_03_l",
    "l_ring1": "ring_01_l",
    "l_ring2": "ring_02_l",
    "l_ring3": "ring_03_l",
    "l_pinky0": "pinky_metacarpal_l",  # SAM3D has this as metacarpal
    "l_pinky1": "pinky_01_l",
    "l_pinky2": "pinky_02_l",
    "l_pinky3": "pinky_03_l",
    # Right fingers
    "r_thumb0": "thumb_01_r",
    "r_thumb1": "thumb_02_r",
    "r_thumb2": "thumb_03_r",
    "r_index1": "index_01_r",
    "r_index2": "index_02_r",
    "r_index3": "index_03_r",
    "r_middle1": "middle_01_r",
    "r_middle2": "middle_02_r",
    "r_middle3": "middle_03_r",
    "r_ring1": "ring_01_r",
    "r_ring2": "ring_02_r",
    "r_ring3": "ring_03_r",
    "r_pinky0": "pinky_metacarpal_r",  # SAM3D has this as metacarpal
    "r_pinky1": "pinky_01_r",
    "r_pinky2": "pinky_02_r",
    "r_pinky3": "pinky_03_r",
}

MH_TO_SAM3D = {v: k for k, v in SAM3D_TO_MH.items()}

# MetaHuman bone hierarchy - defines parent-child with estimated tail offsets
# Each bone: (parent_bone, tail_child_bone or None)
# tail_child_bone is used to determine where the bone should point to
METAHUMAN_HIERARCHY = {
    # Spine chain
    "pelvis": (None, "spine_01"),
    "spine_01": ("pelvis", "spine_02"),
    "spine_02": ("spine_01", "spine_03"),
    "spine_03": ("spine_02", "spine_04"),  # Changed child to spine_04
    "spine_04": ("spine_03", "spine_05"),  # New intermediate bone
    "spine_05": ("spine_04", "neck_01"),   # Changed parent to spine_04
    "neck_01": ("spine_05", "neck_02"),    # Changed child to neck_02
    "neck_02": ("neck_01", "head"),        # New intermediate bone
    "head": ("neck_02", None),             # Changed parent to neck_02
    
    # Left arm
    "clavicle_l": ("spine_05", "upperarm_l"),
    "upperarm_l": ("clavicle_l", "lowerarm_l"),
    "lowerarm_l": ("upperarm_l", "hand_l"),
    "hand_l": ("lowerarm_l", None),        # hand_l is parent to fingers
    
    # Right arm
    "clavicle_r": ("spine_05", "upperarm_r"),
    "upperarm_r": ("clavicle_r", "lowerarm_r"),
    "lowerarm_r": ("upperarm_r", "hand_r"),
    "hand_r": ("lowerarm_r", None),
    
    # Left leg (thigh directly parented to pelvis - standard MetaHuman)
    "thigh_l": ("pelvis", "calf_l"),
    "calf_l": ("thigh_l", "foot_l"),
    "foot_l": ("calf_l", "ball_l"),
    "ball_l": ("foot_l", None),
    
    # Right leg (thigh directly parented to pelvis - standard MetaHuman)
    "thigh_r": ("pelvis", "calf_r"),
    "calf_r": ("thigh_r", "foot_r"),
    "foot_r": ("calf_r", "ball_r"),
    "ball_r": ("foot_r", None),
    
    # Left fingers - with metacarpals where MetaHuman expects them
    "thumb_01_l": ("hand_l", "thumb_02_l"),
    "thumb_02_l": ("thumb_01_l", "thumb_03_l"),
    "thumb_03_l": ("thumb_02_l", None),
    "index_metacarpal_l": ("hand_l", "index_01_l"),
    "index_01_l": ("index_metacarpal_l", "index_02_l"),
    "index_02_l": ("index_01_l", "index_03_l"),
    "index_03_l": ("index_02_l", None),
    "middle_metacarpal_l": ("hand_l", "middle_01_l"),
    "middle_01_l": ("middle_metacarpal_l", "middle_02_l"),
    "middle_02_l": ("middle_01_l", "middle_03_l"),
    "middle_03_l": ("middle_02_l", None),
    "ring_metacarpal_l": ("hand_l", "ring_01_l"),
    "ring_01_l": ("ring_metacarpal_l", "ring_02_l"),
    "ring_02_l": ("ring_01_l", "ring_03_l"),
    "ring_03_l": ("ring_02_l", None),
    "pinky_metacarpal_l": ("hand_l", "pinky_01_l"),
    "pinky_01_l": ("pinky_metacarpal_l", "pinky_02_l"),
    "pinky_02_l": ("pinky_01_l", "pinky_03_l"),
    "pinky_03_l": ("pinky_02_l", None),
    
    # Right fingers - with metacarpals where MetaHuman expects them
    "thumb_01_r": ("hand_r", "thumb_02_r"),
    "thumb_02_r": ("thumb_01_r", "thumb_03_r"),
    "thumb_03_r": ("thumb_02_r", None),
    "index_metacarpal_r": ("hand_r", "index_01_r"),
    "index_01_r": ("index_metacarpal_r", "index_02_r"),
    "index_02_r": ("index_01_r", "index_03_r"),
    "index_03_r": ("index_02_r", None),
    "middle_metacarpal_r": ("hand_r", "middle_01_r"),
    "middle_01_r": ("middle_metacarpal_r", "middle_02_r"),
    "middle_02_r": ("middle_01_r", "middle_03_r"),
    "middle_03_r": ("middle_02_r", None),
    "ring_metacarpal_r": ("hand_r", "ring_01_r"),
    "ring_01_r": ("ring_metacarpal_r", "ring_02_r"),
    "ring_02_r": ("ring_01_r", "ring_03_r"),
    "ring_03_r": ("ring_02_r", None),
    "pinky_metacarpal_r": ("hand_r", "pinky_01_r"),
    "pinky_01_r": ("pinky_metacarpal_r", "pinky_02_r"),
    "pinky_02_r": ("pinky_01_r", "pinky_03_r"),
    "pinky_03_r": ("pinky_02_r", None),
}

# Bones that don't exist in SAM3D but are needed for MetaHuman hierarchy.
# We interpolate them between (start_sam_joint, end_sam_joint).
# Format: "mh_bone_name": ("start_sam_joint", "end_sam_joint", interpolation_weight)
# Weight: 0.0 = at start, 0.5 = midpoint, 1.0 = at end. Default 0.5 if not specified.
INTERPOLATED_BONES = {
    "spine_04": ("c_spine2", "c_spine3"),
    "neck_02": ("c_neck", "c_head"),
    # Metacarpals for index, middle, ring (SAM3D doesn't have these, only pinky0)
    "index_metacarpal_l": ("l_wrist", "l_index1"),
    "index_metacarpal_r": ("r_wrist", "r_index1"),
    "middle_metacarpal_l": ("l_wrist", "l_middle1"),
    "middle_metacarpal_r": ("r_wrist", "r_middle1"),
    "ring_metacarpal_l": ("l_wrist", "l_ring1"),
    "ring_metacarpal_r": ("r_wrist", "r_ring1"),
    # Note: pinky_metacarpal is now mapped directly from l_pinky0/r_pinky0
}


def transform_point(point):
    return COORD_TRANSFORM @ Vector(point)


def cleanup():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for arm in list(bpy.data.armatures):
        bpy.data.armatures.remove(arm)


def create_empties(sam3d_joint_names, rest_positions):
    """Create empties for all MetaHuman bones (including interpolated ones)."""
    joint_index = {n: i for i, n in enumerate(sam3d_joint_names)}
    
    emp_coll = bpy.data.collections.new("Empties")
    bpy.context.scene.collection.children.link(emp_coll)
    
    empties = {}
    
    # 1. Create empties for mapped bones
    for mh_name in METAHUMAN_HIERARCHY.keys():
        if mh_name in INTERPOLATED_BONES:
            continue
            
        sam_name = MH_TO_SAM3D.get(mh_name)
        if not sam_name or sam_name not in joint_index:
            continue
        
        idx = joint_index[sam_name]
        if idx >= len(rest_positions):
            continue
        
        pos = transform_point(rest_positions[idx])
        empty = bpy.data.objects.new("J_" + mh_name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.015
        empty.location = pos
        emp_coll.objects.link(empty)
        empties[mh_name] = empty
        
    # 2. Create empties for interpolated bones
    for mh_name, bone_data in INTERPOLATED_BONES.items():
        # Handle both (start, end) and (start, end, weight) formats
        if len(bone_data) == 3:
            start_sam, end_sam, weight = bone_data
        else:
            start_sam, end_sam = bone_data
            weight = 0.5  # Default: midpoint
        
        if start_sam not in joint_index or end_sam not in joint_index:
            continue
            
        idx1 = joint_index[start_sam]
        idx2 = joint_index[end_sam]
        
        p1 = transform_point(rest_positions[idx1])
        p2 = transform_point(rest_positions[idx2])
        pos = p1 * (1.0 - weight) + p2 * weight
        
        empty = bpy.data.objects.new("J_" + mh_name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.01
        empty.location = pos
        emp_coll.objects.link(empty)
        empties[mh_name] = empty
    
    print("Created " + str(len(empties)) + " empties")
    return empties, joint_index


def animate_empties(empties, frames_data, sam3d_joint_names, joint_index):
    """Animate empties from SAM3D motion data."""
    num_frames = len(frames_data)
    print("Animating " + str(num_frames) + " frames...")
    
    for frame_idx, frame_data in enumerate(frames_data):
        joints = frame_data.get('joints_mhr', frame_data.get('joints3d', []))
        if len(joints) == 127:
            joints = joints[1:]
        
        # 1. Animate mapped bones
        for mh_name, empty in empties.items():
            if mh_name in INTERPOLATED_BONES:
                continue
                
            sam_name = MH_TO_SAM3D.get(mh_name)
            if not sam_name or sam_name not in joint_index:
                continue
            
            idx = joint_index[sam_name]
            if idx >= len(joints):
                continue
            
            pos = transform_point(joints[idx])
            empty.location = pos
            empty.keyframe_insert(data_path="location", frame=frame_idx)
            
        # 2. Animate interpolated bones
        for mh_name, bone_data in INTERPOLATED_BONES.items():
            if mh_name not in empties:
                continue
            
            # Handle both (start, end) and (start, end, weight) formats
            if len(bone_data) == 3:
                start_sam, end_sam, weight = bone_data
            else:
                start_sam, end_sam = bone_data
                weight = 0.5  # Default: midpoint
            
            idx1 = joint_index[start_sam]
            idx2 = joint_index[end_sam]
            
            p1 = transform_point(joints[idx1])
            p2 = transform_point(joints[idx2])
            pos = p1 * (1.0 - weight) + p2 * weight
            
            empties[mh_name].location = pos
            empties[mh_name].keyframe_insert(data_path="location", frame=frame_idx)
        
        if frame_idx % 200 == 0:
            print("  Frame " + str(frame_idx) + "/" + str(num_frames))
    
    return num_frames


def create_armature(empties, sam3d_joint_names, rest_positions, name="MetaHuman_Skeleton"):
    """
    Create armature with MetaHuman-standard bone names and proper hierarchy.
    
    Each bone:
    - Is named like 'pelvis', 'spine_01', etc. (matching MetaHuman)
    - Has proper parent set
    - Has head at its joint position
    - Has tail pointing toward its child (or offset if no child)
    """
    joint_index = {n: i for i, n in enumerate(sam3d_joint_names)}
    
    arm_data = bpy.data.armatures.new(name + "_Armature")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    edit_bones = arm_data.edit_bones
    created_bones = {}
    
    # First pass: create all bones with head positions
    for mh_name, (parent_name, child_name) in METAHUMAN_HIERARCHY.items():
        # Create bone if we have an empty for it (mapped or interpolated)
        if mh_name not in empties:
            continue
        
        bone = edit_bones.new(mh_name)
        
        # Use empty location for head
        # We need to get location from the empty object, but in Edit mode
        # we can't access object location easily if it's animated? 
        # Actually empties are at rest pose on frame 0, and we haven't changed frame yet.
        # But wait, create_empties created them at rest pose. 
        # animate_empties moved them? No, it keyed them.
        # Let's ensure we are at frame 0.
        
        empty = empties[mh_name]
        head_pos = empty.location # This is current location (frame 0 if not played)
        bone.head = head_pos
        
        # Temporary tail - will be set properly in second pass
        bone.tail = head_pos + Vector((0, 0.05, 0))
        
        created_bones[mh_name] = bone
    
    # Second pass: set tails and parents
    for mh_name, (parent_name, child_name) in METAHUMAN_HIERARCHY.items():
        if mh_name not in created_bones:
            continue
        
        bone = created_bones[mh_name]
        
        # Set parent
        if parent_name and parent_name in created_bones:
            bone.parent = created_bones[parent_name]
            bone.use_connect = False  # Don't force head to parent's tail
        
        # Set tail toward child if available
        if child_name and child_name in created_bones:
            child_bone = created_bones[child_name]
            bone.tail = child_bone.head.copy()
        else:
            # No child - create a short offset tail based on bone direction
            if bone.parent:
                direction = bone.head - bone.parent.head
                if direction.length > 0.001:
                    direction.normalize()
                    bone.tail = bone.head + direction * 0.03
                else:
                    bone.tail = bone.head + Vector((0, 0.03, 0))
            else:
                bone.tail = bone.head + Vector((0, 0.05, 0))
        
        # Ensure minimum bone length
        if (bone.tail - bone.head).length < 0.001:
            bone.tail = bone.head + Vector((0, 0.02, 0))
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Add constraints to make bones follow empties
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    
    constrained_count = 0
    
    for mh_name in created_bones.keys():
        if mh_name not in empties:
            continue
        
        pose_bone = arm_obj.pose.bones.get(mh_name)
        if not pose_bone:
            continue
        
        # COPY_LOCATION constraint to follow the empty
        cl = pose_bone.constraints.new('COPY_LOCATION')
        cl.target = empties[mh_name]
        cl.influence = 1.0
        
        # Get child name for STRETCH_TO
        _, child_name = METAHUMAN_HIERARCHY.get(mh_name, (None, None))
        
        if child_name and child_name in empties:
            st = pose_bone.constraints.new('STRETCH_TO')
            st.target = empties[child_name]
            st.rest_length = pose_bone.bone.length
            st.volume = 'NO_VOLUME'
            st.influence = 1.0
        
        constrained_count += 1
    
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()
    
    arm_obj.data.display_type = 'OCTAHEDRAL'
    arm_obj.show_in_front = True
    
    print("Created armature with " + str(len(created_bones)) + " bones")
    print("Constrained " + str(constrained_count) + " bones to empties")
    
    return arm_obj


def bake_animation(arm_obj, num_frames):
    """Bake constraint-driven animation to keyframes for FBX export."""
    print("Baking " + str(num_frames) + " frames...")
    
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    
    bpy.ops.nla.bake(
        frame_start=0,
        frame_end=num_frames - 1,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        bake_types={'POSE'}
    )
    
    bpy.ops.object.mode_set(mode='OBJECT')
    print("Baking complete.")


def add_root_mesh(arm_obj):
    """Add a simple mesh with vertex weights for UE5 import compatibility."""
    # Create a simple cube mesh at the pelvis location
    pelvis_bone = arm_obj.data.bones.get('pelvis')
    if pelvis_bone:
        location = arm_obj.matrix_world @ pelvis_bone.head_local
    else:
        location = (0, 0, 1)  # Default pelvis height
    
    bpy.ops.mesh.primitive_cube_add(size=0.1, location=location)
    mesh_obj = bpy.context.active_object
    mesh_obj.name = "SkeletonMesh"
    
    # Add smooth shading
    bpy.ops.object.shade_smooth()
    
    # Parent to armature
    mesh_obj.parent = arm_obj
    mesh_obj.parent_type = 'OBJECT'
    
    # Create vertex group for pelvis bone and assign all vertices
    vg = mesh_obj.vertex_groups.new(name='pelvis')
    vertices = [v.index for v in mesh_obj.data.vertices]
    vg.add(vertices, 1.0, 'REPLACE')  # Full weight
    
    # Add armature modifier
    mod = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
    mod.object = arm_obj
    mod.use_vertex_groups = True
    
    
    return mesh_obj


def export_fbx(arm_obj, output_path):
    """Export armature with baked animation to FBX for UE5."""
    # Add a small mesh to help UE5 recognize this as a valid skeletal mesh
    mesh_obj = add_root_mesh(arm_obj)
    
    # Select both armature and mesh
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    
    # UE5-compatible FBX export settings
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        object_types={'ARMATURE', 'MESH'},
        # Scale and axes for UE5
        global_scale=100.0,  # Blender meters to UE5 centimeters
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Y',  # Standard UE5 forward axis
        axis_up='Z',
        # Armature settings
        add_leaf_bones=False,
        primary_bone_axis='Y',
        secondary_bone_axis='X',
        armature_nodetype='NULL',
        use_armature_deform_only=True,
        # Animation settings
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,  # No simplification
        # Mesh settings
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',  # Include smoothing groups for UE5
    )
    
    print("Exported FBX to: " + output_path)


def main():
    print("")
    print("=" * 60)
    print("METAHUMAN STANDARD EXPORT")
    print("=" * 60)
    
    cleanup()
    
    print("")
    print("Loading data...")
    print("Motion file: " + MOTION_PATH)
    
    with open(HIERARCHY_PATH, 'r') as f:
        hierarchy = json.load(f)
    with open(MOTION_PATH, 'r') as f:
        motion = json.load(f)
    
    sam3d_joint_names = hierarchy['joints']
    frames_data = motion.get('frames', [motion])
    
    # Skip body_world if present
    if len(sam3d_joint_names) == 127 and sam3d_joint_names[0] in ['body_world', 'Body_World']:
        sam3d_joint_names = sam3d_joint_names[1:]
    
    rest_joints = frames_data[0].get('joints_mhr', frames_data[0].get('joints3d', []))
    if len(rest_joints) == 127:
        rest_joints = rest_joints[1:]
    
    print("SAM3D Joints: " + str(len(sam3d_joint_names)))
    print("Frames: " + str(len(frames_data)))
    
    print("")
    print("[1/4] Creating empties...")
    empties, joint_index = create_empties(sam3d_joint_names, rest_joints)
    
    print("")
    print("[2/4] Animating empties...")
    num_frames = animate_empties(empties, frames_data, sam3d_joint_names, joint_index)
    
    print("")
    print("[3/4] Creating MetaHuman-standard armature...")
    # Ensure we are at frame 0 (rest pose) so empties are in correct position for bone creation
    bpy.context.scene.frame_set(0)
    arm_obj = create_armature(empties, sam3d_joint_names, rest_joints)
    
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1
    # Frame 0 is setting again, which is fine
    bpy.context.scene.frame_set(0)
    
    # Verify skeleton before baking
    print("")
    print("Verifying skeleton...")
    bone_count = len(arm_obj.data.bones)
    print("  Total bones: " + str(bone_count))
    
    # List hierarchy
    root_bones = [b for b in arm_obj.data.bones if b.parent is None]
    print("  Root bones: " + str([b.name for b in root_bones]))
    
    print("")
    print("[4/4] Baking and exporting FBX...")
    bake_animation(arm_obj, num_frames)
    export_fbx(arm_obj, FBX_OUTPUT_PATH)
    
    # Hide empties
    if "Empties" in bpy.data.collections:
        bpy.data.collections["Empties"].hide_viewport = True
    
    # Select armature
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    
    print("")
    print("=" * 60)
    print("DONE!")
    print("")
    print("FBX exported: " + FBX_OUTPUT_PATH)
    print("")
    print("Bone names now match MetaHuman skeleton:")
    print("  pelvis, spine_01, spine_02, spine_03, spine_05")
    print("  clavicle_l, upperarm_l, lowerarm_l, hand_l")
    print("  thigh_l, calf_l, foot_l, ball_l")
    print("  (and corresponding _r for right side)")
    print("")
    print("Next: Import into UE5 and create IK Retargeter")
    print("=" * 60)


if __name__ == "__main__":
    main()
