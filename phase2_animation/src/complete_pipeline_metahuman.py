"""
Complete Pipeline - TRACK TO CHILD
=====================================

Each bone has:
1. COPY_LOCATION to position its HEAD at the joint empty
2. DAMPED_TRACK (or joint rotations) to point its TAIL toward its child's empty

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
FBX_OUTPUT_PATH = os.path.join(BASE_PATH, "data", "metahuman_animation.fbx")

# --- Pipeline Options ---
USE_ROOT_BONE = True
ROOT_BONE_NAME = "root"
ROOT_SOURCE_PREFERENCE = "body_world"
ROOT_STATIC_THRESHOLD = 1e-4

USE_JOINT_ROTATIONS = True
USE_DAMPED_TRACK_FALLBACK = True

REST_POSE_MODE = "frame"  # "frame" or "median"
REST_POSE_FRAME = 0
REST_POSE_MAX_SAMPLES = 200

SET_SCENE_FPS = True
DEFAULT_FPS = 30.0

VALIDATE_AXES_SCALE = True

COORD_TRANSFORM = Matrix([[1,0,0], [0,0,1], [0,-1,0]])
COORD_TRANSFORM_INV = COORD_TRANSFORM.inverted()

BONE_NAME_MAP = {
    "body_world": "root",
    "root": "pelvis", "c_spine0": "spine_01", "c_spine1": "spine_02",
    "c_spine2": "spine_03", "c_spine3": "spine_05", "c_neck": "neck_01", "c_head": "head",
    "l_clavicle": "clavicle_l", "l_uparm": "upperarm_l", "l_lowarm": "lowerarm_l", "l_wrist": "hand_l",
    "r_clavicle": "clavicle_r", "r_uparm": "upperarm_r", "r_lowarm": "lowerarm_r", "r_wrist": "hand_r",
    "l_upleg": "thigh_l", "l_lowleg": "calf_l", "l_foot": "foot_l", "l_ball": "ball_l",
    "r_upleg": "thigh_r", "r_lowleg": "calf_r", "r_foot": "foot_r", "r_ball": "ball_r",
    # Left fingers
    "l_thumb0": "thumb_01_l", "l_thumb1": "thumb_02_l", "l_thumb2": "thumb_03_l", "l_thumb3": "thumb_04_l",
    "l_index1": "index_01_l", "l_index2": "index_02_l", "l_index3": "index_03_l",
    "l_middle1": "middle_01_l", "l_middle2": "middle_02_l", "l_middle3": "middle_03_l",
    "l_ring1": "ring_01_l", "l_ring2": "ring_02_l", "l_ring3": "ring_03_l",
    "l_pinky0": "pinky_01_l", "l_pinky1": "pinky_02_l", "l_pinky2": "pinky_03_l", "l_pinky3": "pinky_04_l",
    # Right fingers
    "r_thumb0": "thumb_01_r", "r_thumb1": "thumb_02_r", "r_thumb2": "thumb_03_r", "r_thumb3": "thumb_04_r",
    "r_index1": "index_01_r", "r_index2": "index_02_r", "r_index3": "index_03_r",
    "r_middle1": "middle_01_r", "r_middle2": "middle_02_r", "r_middle3": "middle_03_r",
    "r_ring1": "ring_01_r", "r_ring2": "ring_02_r", "r_ring3": "ring_03_r",
    "r_pinky0": "pinky_01_r", "r_pinky1": "pinky_02_r", "r_pinky2": "pinky_03_r", "r_pinky3": "pinky_04_r",
}

MAIN_BONES = [
    # Root
    ("root", "pelvis"),
    # Spine
    ("pelvis", "spine_01"), ("spine_01", "spine_02"), ("spine_02", "spine_03"),
    ("spine_03", "spine_05"), ("spine_05", "neck_01"), ("neck_01", "head"),
    # Legs
    ("pelvis", "thigh_l"), ("thigh_l", "calf_l"), ("calf_l", "foot_l"), ("foot_l", "ball_l"),
    ("pelvis", "thigh_r"), ("thigh_r", "calf_r"), ("calf_r", "foot_r"), ("foot_r", "ball_r"),
    # Arms
    ("spine_05", "clavicle_l"), ("clavicle_l", "upperarm_l"), ("upperarm_l", "lowerarm_l"), ("lowerarm_l", "hand_l"),
    ("spine_05", "clavicle_r"), ("clavicle_r", "upperarm_r"), ("upperarm_r", "lowerarm_r"), ("lowerarm_r", "hand_r"),
    # Left fingers
    ("hand_l", "thumb_01_l"), ("thumb_01_l", "thumb_02_l"), ("thumb_02_l", "thumb_03_l"), ("thumb_03_l", "thumb_04_l"),
    ("hand_l", "index_01_l"), ("index_01_l", "index_02_l"), ("index_02_l", "index_03_l"),
    ("hand_l", "middle_01_l"), ("middle_01_l", "middle_02_l"), ("middle_02_l", "middle_03_l"),
    ("hand_l", "ring_01_l"), ("ring_01_l", "ring_02_l"), ("ring_02_l", "ring_03_l"),
    ("hand_l", "pinky_01_l"), ("pinky_01_l", "pinky_02_l"), ("pinky_02_l", "pinky_03_l"), ("pinky_03_l", "pinky_04_l"),
    # Right fingers
    ("hand_r", "thumb_01_r"), ("thumb_01_r", "thumb_02_r"), ("thumb_02_r", "thumb_03_r"), ("thumb_03_r", "thumb_04_r"),
    ("hand_r", "index_01_r"), ("index_01_r", "index_02_r"), ("index_02_r", "index_03_r"),
    ("hand_r", "middle_01_r"), ("middle_01_r", "middle_02_r"), ("middle_02_r", "middle_03_r"),
    ("hand_r", "ring_01_r"), ("ring_01_r", "ring_02_r"), ("ring_02_r", "ring_03_r"),
    ("hand_r", "pinky_01_r"), ("pinky_01_r", "pinky_02_r"), ("pinky_02_r", "pinky_03_r"), ("pinky_03_r", "pinky_04_r"),
]

MH_TO_SAM3D = {v: k for k, v in BONE_NAME_MAP.items()}


def transform_point(point):
    return COORD_TRANSFORM @ Vector(point)


def transform_rotation(rot_matrix):
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


def validate_axes_scale(rest_positions):
    if not rest_positions:
        return
    sam_forward = Vector((0, 0, 1))
    sam_up = Vector((0, 1, 0))
    bl_forward = transform_point(sam_forward)
    bl_up = transform_point(sam_up)
    print("Axis check (SAM3D -> Blender):")
    print("  forward:", tuple(round(v, 3) for v in bl_forward))
    print("  up:", tuple(round(v, 3) for v in bl_up))
    ys = [p[1] for p in rest_positions]
    height = max(ys) - min(ys) if ys else 0.0
    if height < 0.5 or height > 2.5:
        print("WARNING: Unusual skeleton height (~{:.2f}m). Check scale.".format(height))


def get_used_joints():
    used = set()
    for p, c in MAIN_BONES:
        used.add(p)
        used.add(c)
    return used


def create_empties(joint_names, rest_positions, root_source_name=None, root_rest_pos=None):
    joint_index = {n: i for i, n in enumerate(joint_names)}
    used_joints = get_used_joints()

    emp_coll = bpy.data.collections.new("Empties")
    bpy.context.scene.collection.children.link(emp_coll)

    emp_coll.hide_viewport = False
    emp_coll.hide_render = False

    empties = {}
    missing_joints = []

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

    for mh_name in used_joints:
        if mh_name == ROOT_BONE_NAME:
            continue
        sam_name = MH_TO_SAM3D.get(mh_name)
        if not sam_name:
            missing_joints.append((mh_name, "no SAM3D mapping"))
            continue
        if sam_name not in joint_index:
            missing_joints.append((mh_name, sam_name + " not in joint_index"))
            continue
        idx = joint_index[sam_name]
        if idx >= len(rest_positions):
            missing_joints.append((mh_name, "idx " + str(idx) + " >= " + str(len(rest_positions))))
            continue

        pos = to_local(rest_positions[idx])
        empty = bpy.data.objects.new("J_" + mh_name, None)
        empty.empty_display_type = 'SPHERE'
        empty.empty_display_size = 0.015
        if root_empty:
            empty.parent = root_empty
        empty.location = transform_point(pos)
        empty.rotation_mode = 'QUATERNION'
        empty.hide_viewport = False
        emp_coll.objects.link(empty)
        empties[mh_name] = empty

    print("Created " + str(len(empties)) + " empties (expected: " + str(len(used_joints)) + ")")

    print("Created empties:")
    for name in sorted(empties.keys()):
        print("  - " + name)

    if missing_joints:
        print("WARNING: Missing empties for:")
        for name, reason in missing_joints:
            print("  - " + name + ": " + reason)

    return empties, joint_index


def animate_empties(empties, frames_data, joint_names, joint_index, root_source_name=None, root_positions=None, drop_body_world=False):
    num_frames = len(frames_data)
    print("Animating " + str(num_frames) + " frames...")

    used_joints = get_used_joints()
    has_rotations = bool(frames_data and frames_data[0].get('joint_rotations'))

    for frame_idx, frame_data in enumerate(frames_data):
        frame_number = frame_data.get('frame_idx', frame_idx)
        joints = extract_frame_joints(frame_data, drop_body_world)
        rots = extract_frame_rotations(frame_data, drop_body_world) if (has_rotations and USE_JOINT_ROTATIONS) else None

        root_pos = None
        if root_positions and frame_idx < len(root_positions):
            root_pos = root_positions[frame_idx]

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

        for mh_name in used_joints:
            if mh_name == ROOT_BONE_NAME:
                continue
            if mh_name not in empties:
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

            empty = empties[mh_name]
            empty.location = transform_point(pos)
            empty.keyframe_insert(data_path="location", frame=frame_number)

            if rots and idx < len(rots):
                rot_mat = transform_rotation(rots[idx])
                empty.rotation_quaternion = rot_mat.to_quaternion()
                empty.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

        if frame_idx % 200 == 0:
            print("  Frame " + str(frame_idx) + "/" + str(num_frames))

    return num_frames


def create_armature_with_stretch(empties, joint_names, rest_positions, name="MetaHuman_Rig", has_rotations=False):
    joint_index = {n: i for i, n in enumerate(joint_names)}
    used_joints = get_used_joints()
    
    arm_data = bpy.data.armatures.new(name + "_Armature")
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.collection.objects.link(arm_obj)
    
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    edit_bones = arm_data.edit_bones
    bone_map = {}
    
    for parent_name, child_name in MAIN_BONES:
        sam_parent = MH_TO_SAM3D.get(parent_name)
        sam_child = MH_TO_SAM3D.get(child_name)
        
        if not sam_parent or not sam_child:
            continue
        if sam_parent not in joint_index or sam_child not in joint_index:
            continue
            
        p_idx = joint_index[sam_parent]
        c_idx = joint_index[sam_child]
        
        if p_idx >= len(rest_positions) or c_idx >= len(rest_positions):
            continue
        
        bone_name = parent_name + "_to_" + child_name
        bone = edit_bones.new(bone_name)
        bone_map[(parent_name, child_name)] = bone_name  # Store name string, not EditBone object!
        
        head = transform_point(rest_positions[p_idx])
        tail = transform_point(rest_positions[c_idx])
        
        bone.head = head
        bone.tail = tail
        
        if (bone.tail - bone.head).length < 0.001:
            bone.tail = bone.head + Vector((0, 0, 0.02))
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='POSE')
    
    constrained_bones = 0
    missing_parent_empties = []
    missing_child_empties = []
    
    for (parent_name, child_name), bone_name in bone_map.items():
        pose_bone = arm_obj.pose.bones.get(bone_name)
        if not pose_bone:
            continue

        pose_bone.lock_scale = (True, True, True)
        
        has_copy_loc = False
        has_orient = False
        
        if parent_name in empties:
            cl = pose_bone.constraints.new('COPY_LOCATION')
            cl.target = empties[parent_name]
            cl.influence = 1.0  # Ensure full influence
            has_copy_loc = True
        else:
            missing_parent_empties.append((bone_name, parent_name))
        
        if USE_JOINT_ROTATIONS and has_rotations and parent_name in empties:
            cr = pose_bone.constraints.new('COPY_ROTATION')
            cr.target = empties[parent_name]
            cr.mix_mode = 'REPLACE'
            cr.target_space = 'WORLD'
            cr.owner_space = 'WORLD'
            cr.influence = 1.0
            has_orient = True
        elif USE_DAMPED_TRACK_FALLBACK and child_name in empties:
            dt = pose_bone.constraints.new('DAMPED_TRACK')
            dt.target = empties[child_name]
            dt.track_axis = 'TRACK_Y'
            dt.influence = 1.0
            has_orient = True
        else:
            missing_child_empties.append((bone_name, child_name))
        
        if has_copy_loc and has_orient:
            constrained_bones += 1
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Force update to activate constraints
    bpy.context.view_layer.update()
    
    print("Fully constrained bones: " + str(constrained_bones) + "/" + str(len(bone_map)))
    if missing_parent_empties:
        print("WARNING: Missing PARENT empties for " + str(len(missing_parent_empties)) + " bones:")
        for bone_name, parent in missing_parent_empties:
            print("  - " + bone_name + " needs " + parent)
    if missing_child_empties:
        print("WARNING: Missing CHILD empties for " + str(len(missing_child_empties)) + " bones:")
        for bone_name, child in missing_child_empties:
            print("  - " + bone_name + " needs " + child)
    
    arm_obj.data.display_type = 'OCTAHEDRAL'  # Better for seeing constraint status
    arm_obj.show_in_front = True
    
    print("Created armature with " + str(len(bone_map)) + " bones")
    return arm_obj


def bake_animation(arm_obj, num_frames):
    """
    Bake constraint-driven animation to keyframes.
    
    This converts the live constraint evaluation (COPY_LOCATION + DAMPED_TRACK/COPY_ROTATION)
    into actual keyframes on each bone, which is required for FBX export.
    """
    print("Baking " + str(num_frames) + " frames (this may take a minute)...")
    
    # Ensure armature is selected and active
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    
    # Enter Pose mode
    bpy.ops.object.mode_set(mode='POSE')
    
    # Select all pose bones
    bpy.ops.pose.select_all(action='SELECT')
    
    # Bake the animation with visual keying (captures constraint results)
    bpy.ops.nla.bake(
        frame_start=0,
        frame_end=num_frames - 1,
        only_selected=True,
        visual_keying=True,       # Captures constraint results
        clear_constraints=True,   # Remove constraints after baking
        bake_types={'POSE'}       # Bake pose bones
    )
    
    # Return to Object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    print("Baking complete. Constraints removed, keyframes applied.")


def export_fbx(arm_obj, output_path):
    """
    Export the armature with baked animation to FBX format.
    
    Uses UE5-compatible settings for proper import.
    """
    # Ensure only armature is selected
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    
    # Export with UE5-compatible settings
    bpy.ops.export_scene.fbx(
        filepath=output_path,
        use_selection=True,
        object_types={'ARMATURE'},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        global_scale=100.0,
        apply_scale_options='FBX_SCALE_ALL',
        axis_forward='-Y',
        axis_up='Z'
    )
    
    print("Exported FBX to: " + output_path)


def main():
    print("")
    print("=" * 60)
    print("METAHUMAN PIPELINE - TRACK TO CHILD")
    print("=" * 60)
    
    cleanup()
    
    print("")
    print("Loading data...")
    print("Motion file: " + MOTION_PATH)
    with open(HIERARCHY_PATH, 'r') as f:
        hierarchy = json.load(f)
    with open(MOTION_PATH, 'r') as f:
        motion = json.load(f)
    
    joint_names = hierarchy['joints']
    frames_data = motion.get('frames', [motion])

    joint_names, drop_body_world = normalize_joint_names(joint_names)
    rest_joints = select_rest_positions(frames_data, drop_body_world)
    has_rotations = bool(frames_data and frames_data[0].get('joint_rotations'))

    print("Joints: " + str(len(joint_names)) + ", Frames: " + str(len(frames_data)))

    if SET_SCENE_FPS:
        set_scene_fps(motion.get("fps", DEFAULT_FPS))

    if VALIDATE_AXES_SCALE:
        validate_axes_scale(rest_joints)

    joint_index = {n: i for i, n in enumerate(joint_names)}
    root_source_name = choose_root_source(joint_index, frames_data, drop_body_world)
    root_positions = get_root_positions(frames_data, joint_index, root_source_name, drop_body_world)
    root_rest_pos = None
    if root_source_name and root_source_name in joint_index and rest_joints:
        root_rest_pos = rest_joints[joint_index[root_source_name]]

    if root_source_name:
        print("Root source: " + root_source_name)

    print("")
    print("[1/5] Creating empties...")
    empties, joint_index = create_empties(joint_names, rest_joints, root_source_name, root_rest_pos)
    
    print("")
    print("[2/5] Animating empties...")
    num_frames = animate_empties(
        empties,
        frames_data,
        joint_names,
        joint_index,
        root_source_name=root_source_name,
        root_positions=root_positions,
        drop_body_world=drop_body_world,
    )
    
    print("")
    print("[3/5] Creating armature with DAMPED_TRACK...")
    arm_obj = create_armature_with_stretch(empties, joint_names, rest_joints, has_rotations=has_rotations)
    
    max_frame = max(frame.get('frame_idx', idx) for idx, frame in enumerate(frames_data)) if frames_data else 0
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max_frame
    bpy.context.scene.frame_set(0)
    
    print("")
    print("[4/5] Baking animation to keyframes...")
    bake_animation(arm_obj, num_frames)
    
    print("")
    print("[5/5] Exporting FBX...")
    export_fbx(arm_obj, FBX_OUTPUT_PATH)
    
    # Hide empties collection (no longer needed, but kept for reference)
    if "Empties" in bpy.data.collections:
        bpy.data.collections["Empties"].hide_viewport = True
    
    # Select armature for user
    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    
    print("")
    print("=" * 60)
    print("DONE!")
    print("")
    print("FBX exported to: " + FBX_OUTPUT_PATH)
    print("")
    print("Next steps:")
    print("1. Import FBX into UE5")
    print("2. Create IK Rig for imported skeleton")
    print("3. Use IK Retargeter to retarget to MetaHuman")
    print("=" * 60)


if __name__ == "__main__":
    main()
