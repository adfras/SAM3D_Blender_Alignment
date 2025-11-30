# Skeleton Alignment and Fix

- [ ] Analyze current codebase and setup <!-- id: 0 -->
- [ ] Instrument code to record bone/joint positions and rotations <!-- id: 1 -->
- [ ] Analyze recorded data to identify causes of warped fingers and stretched backs <!-- id: 2 -->
- [x] **Fix Stretched Back**: Align SAM3D spine/torso vector to Blender's upright vector.
- [x] **Fix Warped Fingers**:
    - [x] Reorder hand alignment (Step 4) to be before finger loop.
    - [x] Disable re-application of hand rotation after finger alignment.
    - [x] Implement robust rotation matrix for 180-degree flips.
    - [x] Fix double rotation bug in `rotate_branch_around_pivot`.
    - [x] Disable all re-application rotations to prevent overwriting finger alignment.
- [x] **Hierarchical Reconstruction** (New Approach):
    - [x] Implement `reconstruct_skeleton` function in `reset_comparison.py`.
    - [x] Replace `align_full_skeleton` logic with reconstruction.
    - [x] Verify 1:1 alignment with `record_skeleton_data.py`.d: 4 -->
- [ ] Verify alignment with visual checks and data comparison <!-- id: 5 -->
