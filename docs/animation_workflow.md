# Animation Retargeting Workflow

This document explains how to use the SAM3D alignment tools to process and retarget animations.

## 1. Overview

The workflow consists of two main paths:
1.  **Testing/Verification**: Using an FBX file to drive the skeletons and prove compatibility.
2.  **Production**: Using SAM3D inference data (from video) to drive the Blender skeleton.

## 2. Testing with FBX (The "Red Skeleton" Test)

To verify that the SAM3D skeleton (Red) can handle complex animations, we use an existing FBX animation (e.g., `Idle.fbx`).

### Steps:
1.  **Extract Motion**:
    Run the Blender script to convert the FBX to a JSON format our tools can read.
    ```bash
    # Run in terminal (requires Blender installed)
    d:\blender\blender.exe --background --python src/convert_fbx_to_json.py
    ```
    *Output*: `data/idle_motion.json`

2.  **Visualize Side-by-Side**:
    Run the visualization script. This script:
    - Loads the raw motion (Green Skeleton).
    - **Reconstructs** the SAM3D skeleton (Red Skeleton) frame-by-frame using the alignment logic.
    ```bash
    python src/visualize_anim_sidebyside.py
    ```
    *Output*: `animation_sidebyside.gif`

    **Result**: If the Red skeleton matches the Green one, the rig is compatible.

## 3. Production: SAM3D to Blender

To retarget actual video motion to Blender:

1.  **Run Inference**:
    Use `run_sam3d_inference.py` on your video frames to generate `sam3d_data.json`.

2.  **Calculate Rotations**:
    Run the retargeting script. This uses the "Perfect Alignment" logic to compute the bone rotations needed to match the SAM3D pose.
    ```bash
    python src/retarget_motion.py
    ```
    *Output*: `data/animation_data.json`

3.  **Import to Blender**:
    - Open Blender.
    - Select your target Armature (Green Skeleton).
    - Open and run `src/blender_import_anim.py`.
    
    **Result**: Your Blender character will perform the motion captured by SAM3D.

## 4. Key Scripts

- **`src/skeleton_core.py`**: The brain. Contains the `align_full_skeleton` function that ensures the Red skeleton always matches the Blender topology.
- **`src/convert_fbx_to_json.py`**: Helper to get data out of Blender/FBX.
- **`src/retarget_motion.py`**: Calculates the math to transfer motion.

## 5. Troubleshooting & Implementation Details

### Coordinate Systems
- **Blender**: Uses a **Z-up** coordinate system (Right-Handed).
- **SAM3D**: Typically uses Y-up.
- **Our Solution**: We standardized on **Z-up** for all visualization and processing. `skeleton_core.py` was updated to preserve Z-up coordinates during reconstruction.

### Bone Tails
- **Issue**: Without bone tail positions, leaf bones (hands, feet) cannot be reconstructed correctly and appear as collapsed points.
- **Fix**: The `convert_fbx_to_json.py` script exports both `head` and `tail` coordinates for every bone. The reconstruction logic uses the tail to determine the vector for leaf bones.

### Bone Mapping
- **Mixamo vs MHR**: FBX files often use Mixamo naming (e.g., `mixamorig:LeftArm`). These must be mapped to MHR names (e.g., `l_uparm`) during extraction. This mapping is handled in `convert_fbx_to_json.py`.
