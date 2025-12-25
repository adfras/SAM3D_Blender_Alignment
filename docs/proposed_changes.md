# Proposed Changes and Fixes

Date: 2025-12-25

This document lists proposed improvements for the SAM3D -> Blender -> UE5 pipeline. These are not implemented yet.

## Export and Retargeting

1. **Add a dedicated root bone**
   - Use `body_world` as a true root and parent `pelvis` under it.
   - Keeps world translation separate from pelvis motion and improves UE5 root motion workflows.

2. **Lock bone lengths during export**
   - Avoid STRETCH_TO-driven scaling in the final baked FBX.
   - Compute rotations from joint vectors and keep consistent bone lengths for cleaner retargeting.

3. **Use SAM3D joint rotations where available**
   - Apply `joint_rotations` or compute stable local frames to reduce twist artifacts.
   - Especially helpful for forearms, hands, and spine.

4. **Normalize timing and FPS**
   - Set Blender scene FPS based on the source video (or SAM3D metadata).
   - Use `frame_idx` when present to preserve original timing and handle dropped frames.

## Motion Quality

5. **Rest pose selection**
   - Allow choosing a clean rest frame (A/T-pose) or compute a median rest pose.
   - Prevents base-pose offsets in the retargeter.

6. **Contact-aware smoothing**
   - Replace Gaussian-only smoothing with foot locking or contact-aware filters.
   - Reduces foot sliding while keeping upper-body motion smooth.

## Validation and Usability

7. **Axis and scale validation tool**
   - Add a quick check that verifies axes, forward direction, and unit scale before export.
   - Helps prevent backwards arms or scale mismatch in UE5.

8. **Consistent transforms across scripts**
   - Align coordinate transforms in `complete_pipeline_metahuman.py` and `metahuman_standard_export.py`.
   - Reduces confusion when previewing vs exporting.

## Optional (MetaHuman Coverage)

9. **Add twist/IK placeholders (if needed)**
   - MetaHumans include twist and IK bones not present in SAM3D.
   - Optional placeholders can improve retarget stability in some setups.
