# Implementation Plan - Hierarchical Skeleton Reconstruction

The current iterative alignment approach (rotate limb, scale limb) is causing accumulated errors and visual artifacts ("mess"). To resolve this, we will implement a **Hierarchical Reconstruction** approach. Instead of modifying the existing SAM skeleton, we will reconstruct it bone-by-bone to strictly match the Blender skeleton's topology and vectors.

## User Review Required
> [!IMPORTANT]
> This change replaces the entire `align_full_skeleton` logic.
> **Missing Bones**: Bones present in SAM but not in Blender (e.g., twist bones, nulls) will be **collapsed** to their parent's position to prevent visual clutter. This ensures a clean 1:1 match for the main skeleton.

## Proposed Changes

### `reset_comparison.py`

#### [NEW] `reconstruct_skeleton` function
- Input: `sam_joints`, `joints_info` (Blender), `hierarchy`, `sam_names`.
- Logic:
    1.  Align `root` to Blender root.
    2.  Iterate through all joints in topological order.
    3.  For each joint:
        -   If joint and parent exist in Blender:
            -   Calculate vector from Blender (`child - parent`).
            -   Set `sam_joint = sam_parent + blender_vector`.
        -   Else (missing in Blender):
            -   Set `sam_joint = sam_parent` (Collapse).

#### [MODIFY] `align_full_skeleton`
- Replace the entire body of this function with a call to `reconstruct_skeleton`.
- Alternatively, rename `align_full_skeleton` to `align_full_skeleton_old` and create a new wrapper.

## Verification Plan

### Automated Tests
- Run `record_skeleton_data.py`.
- Expect **0.0000** length difference and **0.0000** angle difference for ALL shared bones.
- Verify `debug_bone_vectors.csv`.

### Manual Verification
- Run `reset_comparison.py` to generate `comparison_result.png`.
- Visual check: The Red skeleton should be perfectly superimposed on the Green skeleton for all main bones. Twist bones should be invisible (collapsed).
