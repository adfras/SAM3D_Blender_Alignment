# Walkthrough - Skeleton Alignment Refactoring

I have refactored the skeleton alignment codebase to be cleaner, more maintainable, and reusable.

## Changes

### 1. Created `skeleton_core.py`
This new library contains all the core logic for:
- Loading data (`load_data`)
- Parsing Blender and SAM3D joints (`get_blender_joints`, `get_sam_joints`)
- Mathematical operations (`rotation_matrix_from_vectors`)
- Alignment logic (`rotate_blender_limb`, `initial_sam_alignment`)
- **The "Perfect Alignment" Reconstruction** (`align_full_skeleton`)

### 2. Created `run_alignment.py`
This is the new main entry point. It:
1. Loads data using `skeleton_core`.
2. Performs the alignment pipeline.
3. Saves the output CSVs (`debug_joint_positions.csv`, `debug_bone_vectors.csv`) and a new `aligned_skeleton_data.json`.

### 3. Updated `comparison_sidebyside.py`
Updated to use `skeleton_core` instead of the now-deleted `reset_comparison.py`.

### 4. Cleaned Up
I have archived all legacy scripts, debug logs, and intermediate CSV outputs into the `archive/` folder to keep the workspace clean.

## Project Structure

- **`skeleton_core.py`**: The core library containing all alignment logic.
- **`run_alignment.py`**: The main script to execute the alignment and generate data.
- **`comparison_sidebyside.py`**: Script to visualize the side-by-side comparison.
- **`visualize_superimposed.py`**: Script to visualize the superimposed alignment.
- **`archive/`**: Contains old scripts, debug CSVs, and logs.

## Verification Results

I ran the new `run_alignment.py` and compared its output CSVs with the backups from the old code.
- **Result:** The outputs are **identical**. The refactoring preserved the exact logic of the "perfect alignment".


## Visual Verification

python run_alignment.py
```

To visualize the side-by-side comparison:
```bash
python comparison_sidebyside.py
```

---

## SAM3D Video Inference Pipeline (Added 2025-12-01)

### Overview
Added support for processing video files through SAM3D (Meta's 3D body pose estimation) and visualizing the output as animated skeleton GIFs.

### Changes Made

#### 1. Fixed `run_sam3d_inference.py`
- **Problem**: `NameError: name 'to_list' is not defined` when saving JSON output
- **Solution**: Added `to_list()` helper function after imports to handle JSON serialization of numpy arrays and torch tensors:
```python
def to_list(v):
    """Convert numpy arrays or torch tensors to Python lists for JSON serialization."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().tolist()
    elif isinstance(v, np.ndarray):
        return v.tolist()
    elif isinstance(v, (list, tuple)):
        return [to_list(item) for item in v]
    elif isinstance(v, dict):
        return {k: to_list(val) for k, val in v.items()}
    else:
        return v
```

#### 2. Rewrote `visualize_video_inference.py`
- **Problem**: Original code assumed SMPL 22-joint format, but SAM3D outputs MHR70 (70 joints)
- **Problem**: Skeleton wasn't facing the viewer and looked malformed

**Key fixes:**
1. **Added MHR70 skeleton bone connections** (lines 20-49):
   - Legs: ankle→knee→hip connections (joints 9-14)
   - Torso: shoulder↔hip, shoulder↔shoulder (joints 5-6, 9-10)
   - Arms: shoulder→elbow→wrist (joints 5-8, 41, 62)
   - Head: nose, eyes, ears (joints 0-4)
   - Feet: ankle→toe/heel (joints 13-20)

2. **Fixed coordinate transformation** in `align_joints()`:
   - Center on hip midpoint (joints 9+10) instead of nose for stability
   - SAM3D uses camera coords (X-right, Y-down, Z-forward)
   - Matplotlib 3D uses (X-right, Y-forward, Z-up)
   - Added X-flip to mirror skeleton toward viewer

3. **Updated main() function**:
   - Use `MHR70_BONES` instead of non-existent `SMPL_PARENTS`
   - Bounds checking for joint indices
   - Camera view adjusted (`azim=90`) for front-facing view

### MHR70 Joint Index Reference
Key joint indices from SAM3D output:
- 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
- 5: left_shoulder, 6: right_shoulder
- 7: left_elbow, 8: right_elbow
- 9: left_hip, 10: right_hip
- 11: left_knee, 12: right_knee
- 13: left_ankle, 14: right_ankle
- 15-20: foot keypoints (big_toe, small_toe, heel for each foot)
- 21-61: hand keypoints
- 62: left_wrist, 41: right_wrist
- 69: neck

### Usage
```bash
# Run SAM3D inference on video
python src/run_sam3d_inference.py --image video.mp4 --output data/video_motion.json

# Visualize the output
python src/visualize_video_inference.py --motion data/video_motion.json --output video_inference.gif
```

---

## TODO: SAM3D Inference Improvements

### Visualization Quality
- [x] **Add neck bone connection**: Connect shoulders to neck (joint 69) to head for proper upper body structure
- [x] **Improve head rendering**: Added proper neck→nose connection (joints 5,6→69→0)
- [x] **Add spine visualization**: MHR70 doesn't have explicit spine joints; torso represented by shoulder-hip connections
- [x] **Smoother animation**: Added `--smoothing N` flag for temporal smoothing (default: 5-frame moving average)
- [x] **Better camera angle**: Added `--camera-angle` option: front, side, top, or rotate (auto-rotating view)

### Coordinate System
- [x] **Verify Y-axis direction**: Confirmed SAM3D uses Y-down in camera space; transformation is `-Y → Z-up`
- [x] **Ground plane alignment**: Added automatic ground plane alignment (lowest point at z=0)
- [x] **Scale normalization**: Added `--normalize-height` flag for consistent skeleton height (1.8 units)

### Hand/Finger Visualization
- [x] **Add hand skeleton**: Added `--show-hands` flag to visualize hand keypoints (joints 21-61)
- [x] **Finger bone connections**: Added MHR70_HAND_BONES with full finger articulation for both hands

### Performance
- [ ] **GPU acceleration**: Current inference runs on CPU; enable CUDA for faster processing
- [ ] **Batch processing**: Process multiple frames in parallel
- [ ] **Lower resolution option**: Option to downsample video for faster inference

### Output Formats
- [ ] **BVH export**: Convert motion data to BVH format for animation software
- [ ] **FBX animation**: Direct export to FBX with skeleton animation
- [ ] **Real-time preview**: Show skeleton overlay on original video frames

### Robustness
- [ ] **Handle occlusions**: Improve visualization when joints are occluded/estimated
- [x] **Confidence visualization**: Added `--show-confidence` flag to color-code joints by detection confidence (if available)
- [ ] **Multi-person support**: Handle videos with multiple people

---

## New Command-Line Options (Added 2025-12-02)

```bash
# Full usage
python src/visualize_video_inference.py --motion data/video_motion.json --output output.gif \
    --smoothing 5 \           # Temporal smoothing window (0 to disable)
    --normalize-height \      # Scale skeleton to consistent height
    --show-hands \            # Display hand/finger skeleton
    --show-confidence \       # Color joints by detection confidence
    --camera-angle front \    # front, side, top, or rotate
    --fps 24                  # Output framerate
```
