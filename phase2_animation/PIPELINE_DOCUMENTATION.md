# SAM3D to Blender Animation Pipeline - Complete Documentation

## Overview

This document provides a complete, reproducible guide for the SAM3D to Blender skeleton animation pipeline. It records all decisions made during development and the reasoning behind them.

### Pipeline Summary

```
Video -> SAM3D Inference -> Motion JSON -> (Optional) Smoothing -> Blender (Preview or Export) -> FBX -> UE5 Retarget
```

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Running inference and processing scripts |
| CUDA | 11.x or 12.x | GPU acceleration for SAM3D |
| Blender | 4.0+ | Creating and visualizing animated skeleton |
| SAM3D | Latest | Motion extraction from video |

### SAM3D Installation

SAM3D must be installed in the parent directory:

```bash
cd /path/to/MediaPipeSAM3D/skeleton_alignment_work
git clone https://github.com/facebookresearch/sam-3d-body.git
cd sam-3d-body
pip install -e .
huggingface-cli login  # Required for model weights
```

> **Decision**: SAM3D is installed as a sibling directory rather than a subdirectory to avoid path conflicts with its internal imports.

---

## Pipeline Steps

### Step 1: One-Time Setup - Extract Hierarchy Data

Before processing any video, you must extract the MHR (MetaHuman Rig) skeleton hierarchy from SAM3D.

```bash
cd /path/to/SAM3D_Blender_Alignment/phase2_animation
python src/extract_mhr_hierarchy.py
```

**Output**: `data/mhr_hierarchy.json`

This file contains:
- `joints`: List of 127 joint names (including `body_world` root)
- `parents`: Parent indices for each joint

> **Decision**: The hierarchy is extracted once and stored as JSON because SAM3D's internal skeleton structure is complex and extracting it at runtime adds unnecessary overhead.

### Step 2: Extract Motion from Video

Process a video file to extract 3D joint positions for every frame:

```bash
python src/run_sam3d_inference.py --image path/to/video.mp4 --output data/video_motion_armature.json
```

**Output**: `data/video_motion_armature.json`

This file contains a `frames` array where each frame has:
- `frame_idx`: Frame number
- `joints_mhr`: Array of 127 joint positions [x, y, z]
- `joint_rotations`: Rotation matrices (unused in current pipeline)

> **Decision**: We use `joints_mhr` (MetaHuman Rig positions) rather than `joints3d` (SMPL joints) because MHR provides more detailed finger articulation (127 joints vs 70 joints).

### Step 3: (Optional) Smooth Motion Data

Raw SAM3D output can be noisy. Apply temporal smoothing:

```bash
python src/smooth_motion_data.py
```

**Input**: `data/video_motion_armature.json`
**Output**: `data/video_motion_armature_smooth.json`

The script applies a Gaussian filter to smooth joint positions over time.

> **Decision**: Smoothing is optional but recommended for most videos. The script automatically uses the smoothed version if it exists.

### Step 4: Create Animated Skeleton in Blender (Preview)

Open Blender and run the preview script:

1. Open Blender
2. Switch to the **Scripting** workspace
3. Open `src/complete_pipeline_metahuman.py` via **Text → Open**
4. Press **Alt+P** to run
5. Press **Spacebar** to play the animation

**What the script does:**
1. Creates animated empties for each joint
2. Keyframes empty positions for all frames
3. Creates an armature with bones connecting joints
4. Applies COPY_LOCATION and STRETCH_TO constraints to bones
5. (Optional) Bakes and exports a quick FBX

> **Decision**: We use live constraints (COPY_LOCATION + STRETCH_TO) rather than baking rotation keyframes because:
> - Constraints provide real-time feedback during scrubbing
> - Bones automatically stretch to follow joint positions
> - Avoids complex rotation calculations that can introduce artifacts

### Step 5: Export MetaHuman-Standard FBX from Blender

For UE5 retargeting, use the MetaHuman-standard export script:

1. Open Blender
2. Switch to the **Scripting** workspace
3. Open `src/metahuman_standard_export.py` via **Text → Open**
4. Press **Alt+P** to run

**Output**: `data/metahuman_standard.fbx`

**What the script does:**
1. Creates empties for MetaHuman-standard bone names
2. Interpolates missing bones (spine_04, neck_02, metacarpals)
3. Builds a MetaHuman-compatible hierarchy
4. Bakes animation and exports FBX

### Step 6: Unreal Engine 5 Retargeting

1. Import the FBX as a skeletal mesh with animation
2. Create an IK Rig for the imported skeleton
3. Create an IK Retargeter (source: imported skeleton, target: MetaHuman)
4. Retarget the animation to your MetaHuman character

---

## Key Technical Decisions

### 1. Coordinate System Transformation

SAM3D uses a different coordinate system than Blender:

```python
COORD_TRANSFORM = Matrix([[1,0,0], [0,0,1], [0,-1,0]])
```

This matrix:
- Keeps X axis unchanged
- Swaps Y and Z axes
- Negates the new Y axis

> **Reason**: SAM3D uses Y-up with Z-forward, while Blender uses Z-up with Y-forward. The export script preserves forward direction for UE5 (see `../docs/fix_arm_orientation_walkthrough.md`).

### 2. Joint Name Mapping

SAM3D uses internal joint names (e.g., `l_wrist`), while MetaHuman uses different conventions (e.g., `hand_l`):

```python
SAM3D_TO_MH = {
    "l_wrist": "hand_l",
    "l_thumb0": "thumb_01_l",
    # ... etc
}
```

> **Reason**: MetaHuman naming convention is required for UE5 retargeting. The export script also interpolates missing metacarpals (index/middle/ring) and adds spine_04 and neck_02 to match the MetaHuman hierarchy.

### 3. Bone Definition Using Parent-Child Pairs

Bones are defined as (parent_joint, child_joint) tuples:

```python
MAIN_BONES = [
    ("pelvis", "spine_01"),
    ("spine_01", "spine_02"),
    ("hand_l", "thumb_01_l"),
    # ... etc
]
```

> **Reason**: This approach creates bones that stretch between joints, accurately representing limb segments.

### 4. Storing Bone Names Instead of EditBone Objects

```python
bone_map[(parent_name, child_name)] = bone_name  # String, not EditBone object
```

> **Critical Bug Fix**: EditBone objects become invalid after switching from Edit mode to Object/Pose mode. Accessing them causes UnicodeDecodeError (garbage memory reads). Storing the name string avoids this issue.

### 5. 127 → 126 Joint Handling

The MHR hierarchy has 127 joints, but the first (`body_world`) is a world-space root that should be skipped:

```python
if len(joint_names) == 127 and joint_names[0] == 'body_world':
    joint_names = joint_names[1:]
```

> **Reason**: `body_world` is not a physical joint; it represents the world origin.

---

## File Structure

### Core Pipeline Files

| File | Purpose | When to Run |
|------|---------|-------------|
| `src/extract_mhr_hierarchy.py` | Extract skeleton hierarchy from SAM3D | Once, during setup |
| `src/run_sam3d_inference.py` | Extract motion from video | Per video |
| `src/smooth_motion_data.py` | Smooth motion data | Per video (optional) |
| `src/complete_pipeline_metahuman.py` | Create animated skeleton in Blender | Per video |
| `src/metahuman_standard_export.py` | Export MetaHuman-standard FBX | Per video (for UE5) |

### Data Files

| File | Description |
|------|-------------|
| `data/mhr_hierarchy.json` | Skeleton hierarchy (127 joints) |
| `data/video_motion_armature.json` | Raw motion data from SAM3D |
| `data/video_motion_armature_smooth.json` | Smoothed motion data |
| `data/metahuman_standard.fbx` | MetaHuman-standard FBX export (generated) |

### Archived Files

Files in `src/archive/` and `data/archive/` are previous iterations or experimental approaches that are no longer part of the working pipeline.

---

## Troubleshooting

### Grey Bones in Blender

**Problem**: Some bones appear grey instead of green.

**Cause**: Constraints are not evaluating properly. Possible reasons:
- Target empty doesn't exist
- Target empty is hidden
- Constraint influence is 0

**Solution**: Check console output for "Missing empties" warnings.

### Fingers Not Attached to Hands

**Problem**: Finger bones appear floating near the origin.

**Cause**: Empties for finger joints are not being created or animated.

**Solution**: Ensure joint names in `BONE_NAME_MAP` match exactly with `mhr_hierarchy.json`.

### Arms or Shoulders Facing Backwards in UE5

**Problem**: Arms or shoulders point behind the body after UE5 import.

**Cause**: Incorrect axis conversion or forward direction.

**Solution**: Use `src/metahuman_standard_export.py` and verify the corrected transform settings described in `../docs/fix_arm_orientation_walkthrough.md`.

### UnicodeDecodeError When Running Script

**Problem**: Python throws UnicodeDecodeError with random byte values.

**Cause**: Accessing EditBone objects after mode switch (memory is freed).

**Solution**: Store bone names as strings, not EditBone object references.

---

## Version History

| Date | Change |
|------|--------|
| 2025-12-25 | Added MetaHuman-standard FBX export workflow and UE5 retargeting steps |
| 2025-12-23 | Fixed UnicodeDecodeError bug (EditBone reference after mode switch) |
| 2025-12-23 | Added diagnostic logging for constraint debugging |
| 2025-12-23 | Changed armature display to OCTAHEDRAL for better constraint visibility |
| 2025-12-22 | Initial working pipeline with live constraints |

---

## Future Improvements

1. **Root Motion + FPS Handling**: Add a dedicated root bone and explicit timing control (see `../docs/proposed_changes.md`)
2. **Rotation Use**: Apply SAM3D joint rotations or computed local frames for twist accuracy
3. **Contact-Aware Smoothing**: Reduce foot sliding and preserve contacts
4. **Batch Processing**: Process multiple videos automatically
5. **Real-time Preview**: Add video overlay to verify skeleton tracking accuracy
