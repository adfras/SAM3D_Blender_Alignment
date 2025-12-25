"""
MetaHuman Standard Export
=========================

Creates skeleton with MetaHuman-standard bone names for direct UE5 retargeting.
Bones are named like 'pelvis', 'spine_01', 'thigh_l' instead of 'pelvis_to_spine_01'.

Key differences from complete_pipeline_metahuman.py:
- Bones have single names matching MetaHuman skeleton
- Proper parent-child hierarchy (not just stretch constraints)
- Better compatibility with UE5 IK Retargeter auto-mapping
- Optional root bone separation and joint-rotation usage

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
FBX_GLOBAL_SCALE = 100.0

# --- Pipeline Options ---
USE_ROOT_BONE = True
ROOT_BONE_NAME = "root"
ROOT_SOURCE_PREFERENCE = "body_world"  # "body_world" or "root"
ROOT_STATIC_THRESHOLD = 1e-4  # If body_world is static, fall back to using root

USE_JOINT_ROTATIONS = True
USE_DAMPED_TRACK_FALLBACK = True  # Fallback orientation when rotations missing

REST_POSE_MODE = "frame"  # "frame" or "median"
REST_POSE_FRAME = 0
REST_POSE_MAX_SAMPLES = 200

SET_SCENE_FPS = True
DEFAULT_FPS = 30.0

VALIDATE_AXES_SCALE = True

ADD_TWIST_BONES = False
ADD_IK_BONES = False

COORD_TRANSFORM = Matrix([[1,0,0], [0,0,1], [0,-1,0]])
COORD_TRANSFORM_INV = COORD_TRANSFORM.inverted()

# SAM3D joint name to MetaHuman bone name mapping
SAM3D_TO_MH = {
    "body_world": "root",
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

# Optional twist/IK placeholders (disabled/enabled via flags)
TWIST_BONE_SOURCES = {
    "upperarm_twist_01_l": ("l_uparm", "l_lowarm"),
    "upperarm_twist_01_r": ("r_uparm", "r_lowarm"),
    "lowerarm_twist_01_l": ("l_lowarm", "l_wrist"),
    "lowerarm_twist_01_r": ("r_lowarm", "r_wrist"),
    "thigh_twist_01_l": ("l_upleg", "l_lowleg"),
    "thigh_twist_01_r": ("r_upleg", "r_lowleg"),
    "calf_twist_01_l": ("l_lowleg", "l_foot"),
    "calf_twist_01_r": ("r_lowleg", "r_foot"),
}

IK_BONE_SOURCES = {
    "ik_hand_root": ("root", "c_spine3", 0.5),
    "ik_hand_l": ("l_wrist",),
    "ik_hand_r": ("r_wrist",),
    "ik_foot_root": ("root", "root", 0.0),
    "ik_foot_l": ("l_foot",),
    "ik_foot_r": ("r_foot",),
}

TWIST_BONE_HIERARCHY = {
    "upperarm_twist_01_l": ("upperarm_l", "lowerarm_l"),
    "upperarm_twist_01_r": ("upperarm_r", "lowerarm_r"),
    "lowerarm_twist_01_l": ("lowerarm_l", "hand_l"),
    "lowerarm_twist_01_r": ("lowerarm_r", "hand_r"),
    "thigh_twist_01_l": ("thigh_l", "calf_l"),
    "thigh_twist_01_r": ("thigh_r", "calf_r"),
    "calf_twist_01_l": ("calf_l", "foot_l"),
    "calf_twist_01_r": ("calf_r", "foot_r"),
}

IK_BONE_HIERARCHY = {
    "ik_hand_root": ("root", "ik_hand_l"),
    "ik_hand_l": ("ik_hand_root", None),
    "ik_hand_r": ("ik_hand_root", None),
    "ik_foot_root": ("root", "ik_foot_l"),
    "ik_foot_l": ("ik_foot_root", None),
    "ik_foot_r": ("ik_foot_root", None),
}


def build_metahuman_hierarchy():
    """Build hierarchy with optional root/twist/IK bones."""
    hierarchy = dict(METAHUMAN_HIERARCHY)

    if USE_ROOT_BONE:
        hierarchy[ROOT_BONE_NAME] = (None, "pelvis")
        if "pelvis" in hierarchy:
            parent, child = hierarchy["pelvis"]
            hierarchy["pelvis"] = (ROOT_BONE_NAME, child)
    elif ADD_IK_BONES and ROOT_BONE_NAME not in hierarchy:
        hierarchy[ROOT_BONE_NAME] = (None, "pelvis")

    if ADD_TWIST_BONES:
        hierarchy.update(TWIST_BONE_HIERARCHY)

    if ADD_IK_BONES:
        hierarchy.update(IK_BONE_HIERARCHY)

    return hierarchy


def transform_point(point):
    return COORD_TRANSFORM @ Vector(point)


def transform_rotation(rot_matrix):
    """Convert SAM3D rotation matrix to Blender space."""
    m = Matrix(rot_matrix)
    return COORD_TRANSFORM @ m @ COORD_TRANSFORM_INV


def cleanup():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for arm in list(bpy.data.armatures):
        bpy.data.armatures.remove(arm)


def normalize_joint_names(joint_names):
    drop_body_world = False
    if joint_names and joint_names[0] in ['body_world', 'Body_World'] and not USE_ROOT_BONE:
        joint_names = joint_names[1:]
        drop_body_world = True
    return joint_names, drop_body_world


def extract_frame_joints(frame_data, drop_body_world):
    joints = frame_data.get('joints_mhr', frame_data.get('joints3d', []))
    if drop_body_world and len(joints) == 127:
        joints = joints[1:]
    return joints


def extract_frame_rotations(frame_data, drop_body_world):
    rots = frame_data.get('joint_rotations')
    if rots and drop_body_world and len(rots) == 127:
        rots = rots[1:]
    return rots


def select_rest_positions(frames_data, drop_body_world):
    if not frames_data:
        return []

    if REST_POSE_MODE.lower() == "median":
        sample_count = min(len(frames_data), REST_POSE_MAX_SAMPLES)
        step = max(1, len(frames_data) // sample_count)
        samples = frames_data[::step][:sample_count]
        sample_joints = [extract_frame_joints(frame, drop_body_world) for frame in samples]
        if not sample_joints:
            return []
        n_joints = len(sample_joints[0])
        rest = []
        for j in range(n_joints):
            coords = [sample[j] for sample in sample_joints]
            xs = sorted(c[0] for c in coords)
            ys = sorted(c[1] for c in coords)
            zs = sorted(c[2] for c in coords)
            mid = len(xs) // 2
            if len(xs) % 2 == 1:
                rest.append([xs[mid], ys[mid], zs[mid]])
            else:
                rest.append([
                    0.5 * (xs[mid - 1] + xs[mid]),
                    0.5 * (ys[mid - 1] + ys[mid]),
                    0.5 * (zs[mid - 1] + zs[mid]),
                ])
        return rest

    frame_idx = max(0, min(REST_POSE_FRAME, len(frames_data) - 1))
    return extract_frame_joints(frames_data[frame_idx], drop_body_world)


def set_scene_fps(fps):
    if not fps:
        return
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.render.fps_base = 1.0


def choose_root_source(joint_index, frames_data, drop_body_world):
    if not USE_ROOT_BONE:
        return None

    body_world_idx = joint_index.get("body_world")
    root_idx = joint_index.get("root")

    pref = ROOT_SOURCE_PREFERENCE.lower()

    if pref == "root" and root_idx is not None:
        return "root"

    if body_world_idx is not None:
        positions = []
        for frame in frames_data:
            joints = extract_frame_joints(frame, drop_body_world)
            if body_world_idx < len(joints):
                positions.append(Vector(joints[body_world_idx]))
        if positions:
            max_dev = max((p - positions[0]).length for p in positions)
            if max_dev < ROOT_STATIC_THRESHOLD:
                if pref == "body_world":
                    return "body_world"
                if root_idx is not None:
                    return "root"
        if pref == "body_world":
            return "body_world"
        # Default to body_world if available
        return "body_world"

    if root_idx is not None:
        return "root"

    return None


def get_root_positions(frames_data, joint_index, root_source_name, drop_body_world):
    if not root_source_name:
        return None
    idx = joint_index.get(root_source_name)
    if idx is None:
        return None
    root_positions = []
    for frame in frames_data:
        joints = extract_frame_joints(frame, drop_body_world)
        if idx < len(joints):
            root_positions.append(joints[idx])
    return root_positions


def validate_axes_scale(rest_positions, joint_names):
    if not rest_positions or not joint_names:
        return
    # Axis sanity check
    sam_forward = Vector((0, 0, 1))
    sam_up = Vector((0, 1, 0))
    bl_forward = transform_point(sam_forward)
    bl_up = transform_point(sam_up)
    print("Axis check (SAM3D -> Blender):")
    print("  forward:", tuple(round(v, 3) for v in bl_forward))
    print("  up:", tuple(round(v, 3) for v in bl_up))
    print("  FBX global_scale:", FBX_GLOBAL_SCALE)

    # Scale sanity check: approximate height
    ys = [p[1] for p in rest_positions]
    height = max(ys) - min(ys) if ys else 0.0
    if height < 0.5 or height > 2.5:
        print("WARNING: Unusual skeleton height (~{:.2f}m). Check scale settings.".format(height))


def create_empties(sam3d_joint_names, rest_positions, hierarchy, root_source_name=None, root_rest_pos=None):
    """Create empties for all MetaHuman bones (including interpolated/extra ones)."""
    joint_index = {n: i for i, n in enumerate(sam3d_joint_names)}

    emp_coll = bpy.data.collections.new("Empties")
    bpy.context.scene.collection.children.link(emp_coll)

    empties = {}

    def to_local(pos):
        if root_rest_pos is None:
            return pos
        return [pos[0] - root_rest_pos[0], pos[1] - root_rest_pos[1], pos[2] - root_rest_pos[2]]

    root_empty = None
    if USE_ROOT_BONE and root_source_name and root_rest_pos is not None:
        root_empty = bpy.data.objects.new("J_" + ROOT_BONE_NAME, None)
        root_empty.empty_display_type = 'SPHERE'
        root_empty.empty_display_size = 0.02
        root_empty.location = transform_point(root_rest_pos)
        root_empty.rotation_mode = 'QUATERNION'
        emp_coll.objects.link(root_empty)
        empties[ROOT_BONE_NAME] = root_empty

    extra_bones = set()
    if ADD_TWIST_BONES:
        extra_bones.update(TWIST_BONE_SOURCES.keys())
    if ADD_IK_BONES:
        extra_bones.update(IK_BONE_SOURCES.keys())

    # 1. Create empties for mapped bones
    for mh_name in hierarchy.keys():
        if mh_name == ROOT_BONE_NAME:
            continue
        if mh_name in INTERPOLATED_BONES or mh_name in extra_bones:
            continue

        sam_name = MH_TO_SAM3D.get(mh_name)
        if not sam_name or sam_name not in joint_index:
            continue

        idx = joint_index[sam_name]
        if idx >= len(rest_positions):
            continue

        pos = to_local(rest_positions[idx])
        empty = bpy.data.objects.new("J_" + mh_name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.015
        if root_empty:
            empty.parent = root_empty
        empty.location = transform_point(pos)
        empty.rotation_mode = 'QUATERNION'
        emp_coll.objects.link(empty)
        empties[mh_name] = empty

    # 2. Create empties for interpolated bones
    for mh_name, bone_data in INTERPOLATED_BONES.items():
        if len(bone_data) == 3:
            start_sam, end_sam, weight = bone_data
        else:
            start_sam, end_sam = bone_data
            weight = 0.5

        if start_sam not in joint_index or end_sam not in joint_index:
            continue

        idx1 = joint_index[start_sam]
        idx2 = joint_index[end_sam]

        p1 = Vector(rest_positions[idx1])
        p2 = Vector(rest_positions[idx2])
        pos = p1 * (1.0 - weight) + p2 * weight
        pos = to_local(pos)

        empty = bpy.data.objects.new("J_" + mh_name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.01
        if root_empty:
            empty.parent = root_empty
        empty.location = transform_point(pos)
        empty.rotation_mode = 'QUATERNION'
        emp_coll.objects.link(empty)
        empties[mh_name] = empty

    # 3. Optional extra bones (twist/IK)
    extra_sources = {}
    if ADD_TWIST_BONES:
        extra_sources.update(TWIST_BONE_SOURCES)
    if ADD_IK_BONES:
        extra_sources.update(IK_BONE_SOURCES)

    for mh_name, src in extra_sources.items():
        if mh_name in empties:
            continue
        pos = None
        if len(src) == 1:
            j = src[0]
            if j in joint_index:
                pos = rest_positions[joint_index[j]]
        else:
            start = src[0]
            end = src[1]
            weight = src[2] if len(src) == 3 else 0.5
            if start in joint_index and end in joint_index:
                p1 = Vector(rest_positions[joint_index[start]])
                p2 = Vector(rest_positions[joint_index[end]])
                pos = p1 * (1.0 - weight) + p2 * weight

        if pos is None:
            continue

        pos = to_local(pos)
        empty = bpy.data.objects.new("J_" + mh_name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.01
        if root_empty:
            empty.parent = root_empty
        empty.location = transform_point(pos)
        empty.rotation_mode = 'QUATERNION'
        emp_coll.objects.link(empty)
        empties[mh_name] = empty

    print("Created " + str(len(empties)) + " empties")
    return empties, joint_index


def animate_empties(
    empties,
    frames_data,
    sam3d_joint_names,
    joint_index,
    hierarchy,
    root_source_name=None,
    root_positions=None,
    drop_body_world=False,
    has_rotations=False,
):
    """Animate empties from SAM3D motion data."""
    num_frames = len(frames_data)
    print("Animating " + str(num_frames) + " frames...")

    extra_bones = set()
    if ADD_TWIST_BONES:
        extra_bones.update(TWIST_BONE_SOURCES.keys())
    if ADD_IK_BONES:
        extra_bones.update(IK_BONE_SOURCES.keys())

    for frame_idx, frame_data in enumerate(frames_data):
        frame_number = frame_data.get('frame_idx', frame_idx)
        joints = extract_frame_joints(frame_data, drop_body_world)
        rots = extract_frame_rotations(frame_data, drop_body_world) if (has_rotations and USE_JOINT_ROTATIONS) else None

        root_pos = None
        if root_positions and frame_idx < len(root_positions):
            root_pos = root_positions[frame_idx]

        # Animate root empty
        if USE_ROOT_BONE and root_source_name and ROOT_BONE_NAME in empties and root_pos is not None:
            root_empty = empties[ROOT_BONE_NAME]
            root_empty.location = transform_point(root_pos)
            root_empty.keyframe_insert(data_path="location", frame=frame_number)
            if rots and root_source_name in joint_index:
                r_idx = joint_index[root_source_name]
                if r_idx < len(rots):
                    rot_mat = transform_rotation(rots[r_idx])
                    root_empty.rotation_quaternion = rot_mat.to_quaternion()
                    root_empty.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

        # 1. Animate mapped bones
        for mh_name, empty in empties.items():
            if mh_name == ROOT_BONE_NAME:
                continue
            if mh_name in INTERPOLATED_BONES or mh_name in extra_bones:
                continue

            sam_name = MH_TO_SAM3D.get(mh_name)
            if not sam_name or sam_name not in joint_index:
                continue

            idx = joint_index[sam_name]
            if idx >= len(joints):
                continue

            pos = joints[idx]
            if root_pos is not None:
                pos = [pos[0] - root_pos[0], pos[1] - root_pos[1], pos[2] - root_pos[2]]
            empty.location = transform_point(pos)
            empty.keyframe_insert(data_path="location", frame=frame_number)

            if rots and idx < len(rots):
                rot_mat = transform_rotation(rots[idx])
                empty.rotation_quaternion = rot_mat.to_quaternion()
                empty.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

        # 2. Animate interpolated bones
        for mh_name, bone_data in INTERPOLATED_BONES.items():
            if mh_name not in empties:
                continue

            if len(bone_data) == 3:
                start_sam, end_sam, weight = bone_data
            else:
                start_sam, end_sam = bone_data
                weight = 0.5

            idx1 = joint_index.get(start_sam)
            idx2 = joint_index.get(end_sam)
            if idx1 is None or idx2 is None:
                continue
            if idx1 >= len(joints) or idx2 >= len(joints):
                continue

            p1 = Vector(joints[idx1])
            p2 = Vector(joints[idx2])
            pos = p1 * (1.0 - weight) + p2 * weight
            if root_pos is not None:
                pos = Vector((pos[0] - root_pos[0], pos[1] - root_pos[1], pos[2] - root_pos[2]))

            empties[mh_name].location = transform_point(pos)
            empties[mh_name].keyframe_insert(data_path="location", frame=frame_number)

        # 3. Animate extra bones (twist/IK)
        extra_sources = {}
        if ADD_TWIST_BONES:
            extra_sources.update(TWIST_BONE_SOURCES)
        if ADD_IK_BONES:
            extra_sources.update(IK_BONE_SOURCES)

        for mh_name, src in extra_sources.items():
            if mh_name not in empties:
                continue
            pos = None
            if len(src) == 1:
                j = src[0]
                if j in joint_index and joint_index[j] < len(joints):
                    pos = joints[joint_index[j]]
            else:
                start = src[0]
                end = src[1]
                weight = src[2] if len(src) == 3 else 0.5
                if start in joint_index and end in joint_index:
                    p1 = Vector(joints[joint_index[start]])
                    p2 = Vector(joints[joint_index[end]])
                    pos = p1 * (1.0 - weight) + p2 * weight
            if pos is None:
                continue

            if root_pos is not None:
                pos = [pos[0] - root_pos[0], pos[1] - root_pos[1], pos[2] - root_pos[2]]

            empties[mh_name].location = transform_point(pos)
            empties[mh_name].keyframe_insert(data_path="location", frame=frame_number)

        if frame_idx % 200 == 0:
            print("  Frame " + str(frame_idx) + "/" + str(num_frames))

    return num_frames


def create_armature(empties, hierarchy, name="MetaHuman_Skeleton", has_rotations=False):
    """
    Create armature with MetaHuman-standard bone names and proper hierarchy.

    Each bone:
    - Is named like 'pelvis', 'spine_01', etc. (matching MetaHuman)
    - Has proper parent set
    - Has head at its joint position
    - Has tail pointing toward its child (or offset if no child)
    """
    arm_data = bpy.data.armatures.new(name + "_Armature")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = arm_data.edit_bones
    created_bones = {}

    # First pass: create all bones with head positions
    for mh_name, (parent_name, child_name) in hierarchy.items():
        if mh_name not in empties:
            continue

        bone = edit_bones.new(mh_name)
        empty = empties[mh_name]
        head_pos = empty.location
        bone.head = head_pos
        bone.tail = head_pos + Vector((0, 0.05, 0))
        created_bones[mh_name] = bone

    # Second pass: set tails and parents
    for mh_name, (parent_name, child_name) in hierarchy.items():
        if mh_name not in created_bones:
            continue

        bone = created_bones[mh_name]

        if parent_name and parent_name in created_bones:
            bone.parent = created_bones[parent_name]
            bone.use_connect = False

        if child_name and child_name in created_bones:
            child_bone = created_bones[child_name]
            bone.tail = child_bone.head.copy()
        else:
            if bone.parent:
                direction = bone.head - bone.parent.head
                if direction.length > 0.001:
                    direction.normalize()
                    bone.tail = bone.head + direction * 0.03
                else:
                    bone.tail = bone.head + Vector((0, 0.03, 0))
            else:
                bone.tail = bone.head + Vector((0, 0.05, 0))

        if (bone.tail - bone.head).length < 0.001:
            bone.tail = bone.head + Vector((0, 0.02, 0))

        if ADD_IK_BONES and mh_name in IK_BONE_HIERARCHY:
            bone.use_deform = False

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

        pose_bone.lock_scale = (True, True, True)

        # COPY_LOCATION to follow empty (no scaling)
        cl = pose_bone.constraints.new('COPY_LOCATION')
        cl.target = empties[mh_name]
        cl.influence = 1.0

        # Orientation: use joint rotations if available, otherwise track child
        if USE_JOINT_ROTATIONS and has_rotations and mh_name in MH_TO_SAM3D:
            cr = pose_bone.constraints.new('COPY_ROTATION')
            cr.target = empties[mh_name]
            cr.mix_mode = 'REPLACE'
            cr.target_space = 'WORLD'
            cr.owner_space = 'WORLD'
            cr.influence = 1.0
        elif USE_DAMPED_TRACK_FALLBACK:
            _, child_name = hierarchy.get(mh_name, (None, None))
            if child_name and child_name in empties:
                dt = pose_bone.constraints.new('DAMPED_TRACK')
                dt.target = empties[child_name]
                dt.track_axis = 'TRACK_Y'
                dt.influence = 1.0

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
    root_bone = arm_obj.data.bones.get(ROOT_BONE_NAME) if USE_ROOT_BONE else None
    pelvis_bone = arm_obj.data.bones.get('pelvis')
    if root_bone:
        location = arm_obj.matrix_world @ root_bone.head_local
    elif pelvis_bone:
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
    
    # Create vertex group for root/pelvis bone and assign all vertices
    vg_name = ROOT_BONE_NAME if root_bone else 'pelvis'
    vg = mesh_obj.vertex_groups.new(name=vg_name)
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
    global_scale=FBX_GLOBAL_SCALE,  # Blender meters to UE5 centimeters
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

    sam3d_joint_names, drop_body_world = normalize_joint_names(sam3d_joint_names)
    has_rotations = bool(frames_data and frames_data[0].get('joint_rotations'))

    rest_joints = select_rest_positions(frames_data, drop_body_world)

    print("SAM3D Joints: " + str(len(sam3d_joint_names)))
    print("Frames: " + str(len(frames_data)))

    # Set FPS from motion metadata if available
    if SET_SCENE_FPS:
        set_scene_fps(motion.get("fps", DEFAULT_FPS))

    # Build hierarchy with optional bones
    mh_hierarchy = build_metahuman_hierarchy()

    # Root handling
    joint_index = {n: i for i, n in enumerate(sam3d_joint_names)}
    root_source_name = choose_root_source(joint_index, frames_data, drop_body_world)
    root_positions = get_root_positions(frames_data, joint_index, root_source_name, drop_body_world)
    root_rest_pos = None
    if root_source_name and root_source_name in joint_index and rest_joints:
        root_rest_pos = rest_joints[joint_index[root_source_name]]

    if root_source_name:
        print("Root source: " + root_source_name)

    if VALIDATE_AXES_SCALE:
        validate_axes_scale(rest_joints, sam3d_joint_names)

    print("")
    print("[1/4] Creating empties...")
    empties, joint_index = create_empties(
        sam3d_joint_names,
        rest_joints,
        mh_hierarchy,
        root_source_name=root_source_name,
        root_rest_pos=root_rest_pos,
    )

    print("")
    print("[2/4] Animating empties...")
    num_frames = animate_empties(
        empties,
        frames_data,
        sam3d_joint_names,
        joint_index,
        mh_hierarchy,
        root_source_name=root_source_name,
        root_positions=root_positions,
        drop_body_world=drop_body_world,
        has_rotations=has_rotations,
    )

    print("")
    print("[3/4] Creating MetaHuman-standard armature...")
    bpy.context.scene.frame_set(0)
    arm_obj = create_armature(empties, mh_hierarchy, has_rotations=has_rotations)

    max_frame = max(frame.get('frame_idx', idx) for idx, frame in enumerate(frames_data)) if frames_data else 0
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max_frame
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
    if USE_ROOT_BONE:
        print("  root, pelvis, spine_01, spine_02, spine_03, spine_05")
    else:
        print("  pelvis, spine_01, spine_02, spine_03, spine_05")
    print("  clavicle_l, upperarm_l, lowerarm_l, hand_l")
    print("  thigh_l, calf_l, foot_l, ball_l")
    print("  (and corresponding _r for right side)")
    print("")
    print("Next: Import into UE5 and create IK Retargeter")
    print("=" * 60)


if __name__ == "__main__":
    main()
