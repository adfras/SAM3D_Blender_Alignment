# Phase 2: SAM3D → Blender Skeleton Animation

Extract 3D motion from video using SAM3D and create animated skeletons in Blender with live constraints.

## Overview

This phase builds on Phase 1's alignment work to create a complete **video → animated skeleton** pipeline. The output is a fully rigged skeleton in Blender that animates in real-time following the motion captured from video.

## Pipeline

```
Video → SAM3D Inference → Motion JSON → (Optional) Smoothing → Blender Script → Animated Skeleton
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

### Step 3: Create Animated Skeleton in Blender

1. Open Blender
2. Switch to **Scripting** workspace
3. Open `src/complete_pipeline_metahuman.py`
4. Press **Alt+P** to run
5. Press **Spacebar** to play animation

## Files

| File | Purpose |
|------|---------|
| `src/run_sam3d_inference.py` | Extract motion from video |
| `src/smooth_motion_data.py` | Apply temporal smoothing |
| `src/complete_pipeline_metahuman.py` | Create animated skeleton in Blender |
| `src/extract_mhr_hierarchy.py` | Extract MHR skeleton (run once) |
| `data/mhr_hierarchy.json` | Skeleton hierarchy (127 joints) |

## Technical Details

See [PIPELINE_DOCUMENTATION.md](PIPELINE_DOCUMENTATION.md) for:
- Coordinate system transformations
- Joint name mappings (SAM3D → MetaHuman)
- Constraint setup (COPY_LOCATION + STRETCH_TO)
- Troubleshooting guide

## Key Technical Decisions

1. **Live Constraints**: Using COPY_LOCATION + STRETCH_TO instead of baked rotations for real-time feedback
2. **Position-based Animation**: Animating joint positions directly rather than computing complex rotation transforms
3. **MetaHuman Naming**: Using MetaHuman bone naming convention for eventual UE5 retargeting compatibility
