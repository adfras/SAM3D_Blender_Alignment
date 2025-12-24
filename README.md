# SAM3D to Blender: Skeleton Alignment & Animation Pipeline

A complete pipeline for extracting 3D motion from video using SAM3D and creating animated skeletons in Blender.

## Project Journey

This project evolved through two phases:

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Skeleton Alignment | ✅ Complete |
| **Phase 2** | Skeleton Animation | ✅ Complete |

### Phase 1: Skeleton Alignment (December 2025)

**Goal**: Align SAM3D skeleton topology with Blender's MHR armature.

**Approach**: Hierarchical reconstruction that rebuilds the SAM3D skeleton bone-by-bone using Blender's rest pose vectors. This ensures 1:1 compatibility for animation retargeting.

**Key Achievement**: Zero rotational or scaling errors between source and target skeletons.

📁 See [phase1_alignment/](phase1_alignment/) for the alignment tools.

---

### Phase 2: Skeleton Animation (December 2025)

**Goal**: Create fully animated skeletons in Blender from video input.

**Approach**: Extract 127-joint MHR positions from SAM3D, animate empties at each joint, and use live constraints (COPY_LOCATION + STRETCH_TO) to drive an armature.

**Key Achievements**:
- Real-time constraint-driven animation
- Full finger articulation (all 10 fingers)
- MetaHuman-compatible bone naming
- Temporal smoothing for noise reduction

📁 See [phase2_animation/](phase2_animation/) for the animation pipeline.

---

## Quick Start (Phase 2 Pipeline)

### Prerequisites

- Python 3.10+ with CUDA
- Blender 4.0+
- [SAM3D](https://github.com/facebookresearch/sam-3d-body) installed

### Pipeline

```
Video → SAM3D → JSON → Blender → Animated Skeleton
```

### Usage

```bash
# 1. Extract motion from video
python phase2_animation/src/run_sam3d_inference.py --image video.mp4 --output phase2_animation/data/video_motion.json

# 2. (Optional) Smooth the data
python phase2_animation/src/smooth_motion_data.py

# 3. Open Blender and run the script
# In Blender: Text → Open → phase2_animation/src/complete_pipeline_metahuman.py
# Press Alt+P to run, Spacebar to play
```

## Project Structure

```
SAM3D_Blender_Alignment/
│
├── phase1_alignment/           # Skeleton alignment tools
│   ├── src/
│   │   ├── skeleton_core.py
│   │   ├── run_alignment.py
│   │   ├── comparison_sidebyside.py
│   │   └── visualize_superimposed.py
│   ├── data/
│   └── README.md
│
├── phase2_animation/           # Skeleton animation pipeline
│   ├── src/
│   │   ├── run_sam3d_inference.py
│   │   ├── smooth_motion_data.py
│   │   ├── complete_pipeline_metahuman.py
│   │   └── extract_mhr_hierarchy.py
│   ├── data/
│   │   └── mhr_hierarchy.json
│   ├── PIPELINE_DOCUMENTATION.md
│   └── README.md
│
├── docs/                       # Documentation
├── requirements.txt
└── README.md                   # This file
```

## Documentation

- **Phase 1**: [phase1_alignment/README.md](phase1_alignment/README.md)
- **Phase 2**: [phase2_animation/README.md](phase2_animation/README.md)
- **Phase 2 Technical Details**: [phase2_animation/PIPELINE_DOCUMENTATION.md](phase2_animation/PIPELINE_DOCUMENTATION.md)
- **Project Docs**: [docs/](docs/)

## Requirements

Install dependencies for both phases:

```bash
pip install -r requirements.txt
```

## Future Work

- [ ] FBX export for Unreal Engine import
- [ ] IK Retargeter integration for MetaHuman
- [ ] Batch video processing
- [ ] Real-time preview overlay

---

*This project documents the journey of building a video-to-MetaHuman animation pipeline using SAM3D and Blender.*
