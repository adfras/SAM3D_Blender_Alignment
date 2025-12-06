# SAM3D Skeleton in Blender - Setup Guide

## Overview

This document explains how to import and visualize SAM3D motion capture data as an animated red skeleton in Blender.

## Data Format: MHR70

The SAM3D video inference outputs joint positions in **MHR70 format**:
- 70 joints per frame
- Stored in `data/video_motion_full.json`
- Structure: `{"frames": [{"frame_idx": 0, "joints3d": [[x,y,z], ...]}, ...]}`

### Key Joint Indices
| Index | Joint Name |
|-------|------------|
| 0 | nose |
| 5 | left_shoulder |
| 6 | right_shoulder |
| 7 | left_elbow |
| 8 | right_elbow |
| 9 | left_hip |
| 10 | right_hip |
| 11 | left_knee |
| 12 | right_knee |
| 13 | left_ankle |
| 14 | right_ankle |
| 41 | right_wrist |
| 62 | left_wrist |
| 69 | neck |

### Bone Connections
```python
MHR70_BONES = [
    (13, 11), (11, 9),   # left leg
    (14, 12), (12, 10),  # right leg
    (9, 10),             # pelvis
    (5, 9), (6, 10),     # torso sides
    (5, 6),              # shoulders
    (5, 69), (6, 69),    # neck
    (69, 0),             # head
    (5, 7), (7, 62),     # left arm
    (6, 8), (8, 41),     # right arm
]
```

## Coordinate System Conversion

SAM3D and Blender use different coordinate systems:

| System | X | Y | Z |
|--------|---|---|---|
| SAM3D | Right | Down | Forward |
| Blender | Right | Forward | Up |

### Conversion Formula
```python
def align_joints(joints):
    # Center on hip midpoint
    root = (joints[9] + joints[10]) / 2.0
    centered = joints - root

    # Convert coordinates
    rotated = np.zeros_like(centered)
    rotated[:, 0] = centered[:, 0]       # X stays
    rotated[:, 1] = -centered[:, 2]      # Y = -Z (depth flipped)
    rotated[:, 2] = -centered[:, 1]      # Z = -Y (up)

    # Ground align (feet at z=0)
    min_z = np.min(rotated[:, 2])
    rotated[:, 2] -= min_z

    return rotated
```

This is the **same transform** used in `visualize_video_inference.py` for matplotlib.

## Animation Method: Shape Keys

After testing several approaches, **shape keys** proved most reliable:

### Why Shape Keys?
- ❌ Armature rotations: Coordinate system issues, mangled poses
- ❌ Hook modifiers: Don't follow animated empties properly
- ✅ Shape keys: Direct vertex position animation, always works

### How It Works
1. Create a single mesh with 14 vertices (one per joint) and 15 edges (bones)
2. Add a "Basis" shape key (rest pose)
3. For each frame, create a shape key with the exact vertex positions
4. Keyframe each shape key: value=1 at its frame, value=0 otherwise
5. Use CONSTANT interpolation (no blending between frames)

## Usage Instructions

### Prerequisites
- Blender 4.0+ (tested on 5.0)
- Motion data in `data/video_motion_full.json`

### Steps
1. Open Blender
2. Go to **Scripting** tab
3. Click **Open** and select:
   ```
   D:\MediaPipeSAM3D\skeleton_alignment_work\src\blender_sam3d_direct.py
   ```
4. Press **Alt+P** to run
5. Press **Space** to play animation

### What Gets Created
- `SAM3D_Skeleton` mesh object
- Red emissive material
- Skin modifier for bone thickness
- Shape keys for each frame (741 frames = 741 shape keys)

## Troubleshooting

### Skeleton looks wrong/mangled
- Ensure using `video_motion_full.json` (MHR70 format)
- Check coordinate conversion matches matplotlib

### Animation doesn't play
- Verify timeline is set (frame 0 to N-1)
- Check shape keys exist in Object Data Properties > Shape Keys

### Bones don't move
- Shape key interpolation must be CONSTANT, not LINEAR
- Each frame's shape key should be value=1 only at that frame

## File References

| File | Purpose |
|------|---------|
| `src/blender_sam3d_direct.py` | Blender import script |
| `src/visualize_video_inference.py` | Matplotlib reference (same coord system) |
| `data/video_motion_full.json` | Motion capture data (741 frames, 70 joints) |
| `data/mhr_hierarchy.json` | Joint hierarchy (not used for MHR70 visualization) |

## Technical Notes

### Why Not Use mhr_hierarchy.json?
The `mhr_hierarchy.json` defines a different skeleton with named joints (root, l_upleg, c_spine0, etc.) - this is for **retargeting to character rigs**, not for direct visualization of SAM3D output.

For visualization, use the **MHR70 numerical indices** directly, which match the `joints3d` array order.

### Performance
- 741 frames creates 741 shape keys
- Import takes ~30 seconds
- Playback is smooth in Blender viewport
