"""
SAM3D Video Inference Pipeline

Processes video input to extract 3D skeleton and mesh data using SAM3D model.

Usage:
    python src/run_sam3d_inference.py --image video.mp4 --output data/video_motion_armature.json --save-mesh

Arguments:
    --image         Input video file (mp4, avi, mov, mkv) or image
    --output        Output JSON file path
    --save-mesh     Include mesh vertices in output (larger file)
    --skip-frames   Process every Nth frame (default: 1 = all frames)

Output JSON structure:
    {
        "frames": [
            {
                "frame_idx": 0,
                "joints3d": [...],           # Simplified 70-joint skeleton
                "joints_mhr": [...],         # Full 127-joint MHR skeleton
                "joint_rotations": [...],    # 127 x 3x3 rotation matrices
                "vertices": [...]            # Optional: 18439 mesh vertices
            },
            ...
        ]
        "fps": 30.0,
        "frame_stride": 1,
        "source_total_frames": 900,
        "source_path": "video.mp4"
    }

Prerequisites:
    - sam-3d-body repository cloned to ../sam-3d-body/
    - PyTorch with CUDA (recommended for performance)
    - Model weights will be downloaded from Hugging Face on first run
"""

import argparse
import os
import cv2
import numpy as np
import torch
import json
import sys
import time

# Add sam-3d-body to path
SAM3D_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "sam-3d-body")
sys.path.append(SAM3D_PATH)

try:
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
except ImportError:
    print("ERROR: Could not import sam_3d_body.")
    print(f"Ensure sam-3d-body is cloned at: {SAM3D_PATH}")
    print("Run: git clone https://github.com/facebookresearch/sam-3d-body.git")
    sys.exit(1)


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


def process_video(estimator, video_path, save_mesh=False, skip_frames=1):
    """Process video and extract motion data."""
    print(f"Processing video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total_frames} frames at {fps:.1f} fps")
    
    if skip_frames > 1:
        print(f"Processing every {skip_frames} frame(s) ({total_frames // skip_frames} frames)")

    frames_data = []
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames if requested
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        with torch.inference_mode():
            outputs = estimator.process_one_image(frame_rgb)
        
        if outputs:
            out_data = outputs[0]  # First person detected

            # Debug: Print available keys on first frame
            if frame_count == 0:
                print(f"SAM3D output keys: {list(out_data.keys())}")
                if 'pred_vertices' in out_data:
                    verts = out_data['pred_vertices']
                    if hasattr(verts, 'shape'):
                        print(f"Mesh vertices shape: {verts.shape}")
                if 'pred_joint_coords' in out_data:
                    jc = out_data['pred_joint_coords']
                    if hasattr(jc, 'shape'):
                        print(f"MHR joints shape: {jc.shape}")
                if 'pred_global_rots' in out_data:
                    rots = out_data['pred_global_rots']
                    if hasattr(rots, 'shape'):
                        print(f"Joint rotations shape: {rots.shape}")
                # SMPL pose parameters
                if 'body_pose_params' in out_data:
                    bp = out_data['body_pose_params']
                    if hasattr(bp, 'shape'):
                        print(f"Body pose params shape: {bp.shape}")
                if 'hand_pose_params' in out_data:
                    hp = out_data['hand_pose_params']
                    if hasattr(hp, 'shape'):
                        print(f"Hand pose params shape: {hp.shape}")
                if 'shape_params' in out_data:
                    sp = out_data['shape_params']
                    if hasattr(sp, 'shape'):
                        print(f"Shape params shape: {sp.shape}")

            # Extract joints
            joints = None
            joints_full = None

            if 'pred_keypoints_3d' in out_data:
                joints = out_data['pred_keypoints_3d']
            elif 'joints3d' in out_data:
                joints = out_data['joints3d']

            if 'pred_joint_coords' in out_data:
                joints_full = out_data['pred_joint_coords']
                if joints is None:
                    joints = joints_full

            if joints is not None:
                if isinstance(joints, (np.ndarray, torch.Tensor)):
                    joints = to_list(joints)

                frame_entry = {
                    "frame_idx": frame_count,
                    "joints3d": joints
                }

                # Full 127-joint skeleton
                if joints_full is not None:
                    frame_entry["joints_mhr"] = to_list(joints_full)

                # Joint rotations (global rotation matrices)
                if 'pred_global_rots' in out_data:
                    frame_entry["joint_rotations"] = to_list(out_data['pred_global_rots'])

                # SMPL pose parameters (axis-angle format) - CRITICAL for natural animation
                if 'body_pose_params' in out_data:
                    frame_entry["body_pose"] = to_list(out_data['body_pose_params'])
                
                if 'hand_pose_params' in out_data:
                    frame_entry["hand_pose"] = to_list(out_data['hand_pose_params'])
                
                if 'shape_params' in out_data:
                    frame_entry["shape_params"] = to_list(out_data['shape_params'])
                
                # Camera/root translation
                if 'pred_cam_t' in out_data:
                    frame_entry["camera_translation"] = to_list(out_data['pred_cam_t'])
                
                # Global orientation (pelvis rotation)
                if 'global_orient' in out_data:
                    frame_entry["global_orient"] = to_list(out_data['global_orient'])

                # Mesh vertices
                if save_mesh and 'pred_vertices' in out_data:
                    frame_entry["vertices"] = to_list(out_data['pred_vertices'])

                frames_data.append(frame_entry)
                processed_count += 1
            else:
                print(f"Frame {frame_count}: No joints found")
        else:
            print(f"Frame {frame_count}: No person detected")

        frame_count += 1

        # Progress update
        if processed_count > 0 and processed_count % 50 == 0:
            elapsed = time.time() - start_time
            fps_proc = processed_count / elapsed
            remaining = (total_frames // skip_frames - processed_count) / fps_proc
            print(f"Processed {processed_count} frames... ({fps_proc:.1f} fps, ~{remaining:.0f}s remaining)")
            
    cap.release()

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s ({processed_count / total_time:.1f} fps)")

    return {
        "frames": frames_data,
        "fps": float(fps) if fps else None,
        "frame_stride": int(skip_frames),
        "source_total_frames": int(total_frames),
        "source_path": video_path,
    }


def process_image(estimator, image_path, save_mesh=False):
    """Process single image."""
    print(f"Processing image: {image_path}")
    outputs = estimator.process_one_image(image_path)

    if not outputs:
        print("No person detected.")
        return None

    out_data = outputs[0]
    return {
        **{k: to_list(v) for k, v in out_data.items()},
        "fps": None,
        "frame_stride": 1,
        "source_total_frames": 1,
        "source_path": image_path,
    }


def main():
    parser = argparse.ArgumentParser(description="SAM3D Video Inference Pipeline")
    parser.add_argument("--image", required=True, help="Input video or image file")
    parser.add_argument("--output", default="data/video_motion_armature.json", help="Output JSON path")
    parser.add_argument("--repo_id", default="facebook/sam-3d-body-dinov3", help="Hugging Face model ID")
    parser.add_argument("--save-mesh", action="store_true", help="Include mesh vertices (larger file)")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load model
    print(f"Loading SAM3D model from Hugging Face: {args.repo_id}...")
    try:
        from sam_3d_body import load_sam_3d_body_hf
        model, model_cfg = load_sam_3d_body_hf(args.repo_id, device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure you have access and are logged in: huggingface-cli login")
        return

    model = model.to(device)
    model.eval()

    # Move encoder to device
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
        model.backbone.encoder = model.backbone.encoder.to(device)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,
        human_segmentor=None,
        fov_estimator=None,
    )

    # Check if input is video
    is_video = args.image.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        result = process_video(estimator, args.image, args.save_mesh, args.skip_frames)
    else:
        result = process_image(estimator, args.image, args.save_mesh)

    if result:
        # Ensure output directory exists
        output_path = os.path.join(os.path.dirname(__file__), "..", args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Saved to {output_path}")
        if is_video:
            print(f"Total frames: {len(result.get('frames', []))}")


if __name__ == "__main__":
    main()
