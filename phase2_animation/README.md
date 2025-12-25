# Phase 2: SAM3D → Blender Skeleton Animation

Extract 3D motion from video using SAM3D and create animated skeletons in Blender with live constraints.

## Overview

This phase builds on Phase 1's alignment work to create a complete **video → animated skeleton** pipeline. The output is a fully rigged skeleton in Blender that animates in real-time following the motion captured from video.

## Pipeline

```
Video -> SAM3D Inference -> Motion JSON -> (Optional) Smoothing -> Blender (Preview or Export) -> FBX -> UE5 Retarget
```

## Quick Start

### Step 1: Extract Motion from Video

```bash
python src/run_sam3d_inference.py --image video.mp4 --output data/video_motion_armature.json
```

### Step 2: (Optional) Smooth Motion Data

```bash
python src/smooth_motion_data.py
```

Optional flags:
- `--contact-aware` to reduce foot sliding
- `--normalize-lengths` to enforce fixed bone lengths
- `--fps` to override FPS (defaults to value stored in the input JSON)

### Step 3: Create Animated Skeleton in Blender

1. Open Blender
2. Switch to **Scripting** workspace
3. Open `src/complete_pipeline_metahuman.py` for preview, or `src/metahuman_standard_export.py` for UE5 export
4. Press **Alt+P** to run
5. Press **Spacebar** to play animation (preview script)

**FBX Output (export script)**: `data/metahuman_standard.fbx`

Export features:
- Optional root bone for root motion workflows
- Joint-rotation usage (if present in SAM3D output)
- Axis/scale validation before export
- Optional twist/IK placeholders (toggle in `metahuman_standard_export.py`)
- Rest pose selection (frame or median) via script constants

### Step 4: Unreal Engine 5 Retargeting

1. Import the FBX as a skeletal mesh with animation
2. Create an IK Rig for the imported skeleton
3. Create an IK Retargeter (source: imported skeleton, target: MetaHuman)
4. Retarget the animation to your MetaHuman character

## Files

| File | Purpose |
|------|---------|
| `src/run_sam3d_inference.py` | Extract motion from video |
| `src/smooth_motion_data.py` | Apply temporal smoothing |
| `src/complete_pipeline_metahuman.py` | Create animated skeleton in Blender |
| `src/metahuman_standard_export.py` | Export MetaHuman-standard FBX |
| `src/extract_mhr_hierarchy.py` | Extract MHR skeleton (run once) |
| `data/mhr_hierarchy.json` | Skeleton hierarchy (127 joints) |

## Technical Details

See [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md) for:
- Coordinate system transformations
- Joint name mappings (SAM3D → MetaHuman)
- Constraint setup (COPY_LOCATION + DAMPED_TRACK/COPY_ROTATION)
- UE5 export settings and axis corrections (see `../docs/fix_arm_orientation_walkthrough.md`)
- Troubleshooting guide

## Key Technical Decisions

1. **Live Constraints**: Using COPY_LOCATION + DAMPED_TRACK/COPY_ROTATION instead of baked rotations for real-time feedback
2. **Position-based Animation**: Animating joint positions directly rather than computing complex rotation transforms
3. **MetaHuman Naming**: Using MetaHuman bone naming convention for eventual UE5 retargeting compatibility
