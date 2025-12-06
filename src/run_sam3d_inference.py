import argparse
import os
import cv2
import numpy as np
import torch
import json
import sys

# Add sam-3d-body to path
# Add sam-3d-body to path (it is in the root, one level up from src)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "sam-3d-body"))

try:
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
except ImportError:
    print("Could not import sam_3d_body. Make sure it is installed or in the python path.")
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="qYwLO.png")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--repo_id", default="facebook/sam-3d-body-dinov3", help="Hugging Face Repo ID")
    parser.add_argument("--output", default="sam3d_data.json")
    parser.add_argument("--mhr_path", default="")
    parser.add_argument("--save-mesh", action="store_true", help="Save mesh vertices (larger file)")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every Nth frame (default: 1 = all frames)")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Enable TF32 for faster inference on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Load Model
    try:
        if args.checkpoint:
            print(f"Loading model from {args.checkpoint}...")
            model, model_cfg = load_sam_3d_body(args.checkpoint, device=device, mhr_path=args.mhr_path)
        else:
            print(f"Loading model from Hugging Face {args.repo_id}...")
            from sam_3d_body import load_sam_3d_body_hf
            model, model_cfg = load_sam_3d_body_hf(args.repo_id, device=device)
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If using Hugging Face, ensure you have access and are logged in (huggingface-cli login).")
        return

    # Ensure model and ALL submodules are on the correct device
    # The model.to(device) in build_models.py doesn't always move torch.hub loaded modules
    model = model.to(device)
    model.eval()

    # Explicitly move the DINOv3 encoder using its custom helper (handles stray tensors)
    if hasattr(model, 'backbone') and hasattr(model.backbone, '_move_encoder_to_device'):
        model.backbone._move_encoder_to_device(device)
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'encoder'):
        model.backbone.encoder = model.backbone.encoder.to(device)
        print(f"Moved backbone encoder to {device}")

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None, # Assuming single person or center crop
        human_segmentor=None,
        fov_estimator=None,
    )

    # Check if input is video
    is_video = args.image.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

    if is_video:
        import time
        print(f"Processing video {args.image}...")
        cap = cv2.VideoCapture(args.image)
        if not cap.isOpened():
            print(f"Error opening video {args.image}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {total_frames} frames at {fps:.1f} fps")
        if args.skip_frames > 1:
            print(f"Processing every {args.skip_frames} frame(s) ({total_frames // args.skip_frames} frames)")

        frames_data = []
        frame_count = 0
        processed_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if requested
            if frame_count % args.skip_frames != 0:
                frame_count += 1
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            # estimator.process_one_image accepts numpy array (H, W, 3)
            with torch.inference_mode():  # Faster than torch.no_grad()
                outputs = estimator.process_one_image(frame_rgb)
            
            if outputs:
                # Take first person detected
                out_data = outputs[0]

                # Extract keypoints (usually 'pred_keypoints_3d' or 'pred_joint_coords')
                # Map to 'joints3d' for skeleton_core compatibility
                joints = None
                if 'pred_keypoints_3d' in out_data:
                    joints = out_data['pred_keypoints_3d']
                elif 'pred_joint_coords' in out_data:
                    joints = out_data['pred_joint_coords']
                elif 'joints3d' in out_data:
                    joints = out_data['joints3d']

                if joints is not None:
                    # Convert to list
                    if isinstance(joints, (np.ndarray, torch.Tensor)):
                         joints = to_list(joints)

                    frame_entry = {
                        "frame_idx": frame_count,
                        "joints3d": joints
                    }

                    # Also save mesh data if available and requested
                    if args.save_mesh and 'pred_vertices' in out_data:
                        frame_entry["vertices"] = to_list(out_data['pred_vertices'])

                    # Save SMPL parameters if available
                    if 'pred_smpl_params' in out_data:
                        frame_entry["smpl_params"] = to_list(out_data['pred_smpl_params'])

                    # Save individual SMPL components if available
                    for key in ['body_pose', 'global_orient', 'betas', 'transl']:
                        if key in out_data:
                            frame_entry[key] = to_list(out_data[key])

                    # Save camera if available
                    if 'pred_camera' in out_data:
                        frame_entry["camera"] = to_list(out_data['pred_camera'])

                    frames_data.append(frame_entry)
                    processed_count += 1
                else:
                    print(f"Frame {frame_count}: No joints found in output keys: {out_data.keys()}")
            else:
                print(f"Frame {frame_count}: No person detected.")

            frame_count += 1

            # Progress update with timing
            if processed_count > 0 and processed_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_proc = processed_count / elapsed
                remaining = (total_frames // args.skip_frames - processed_count) / fps_proc
                print(f"Processed {processed_count} frames... ({fps_proc:.1f} fps, ~{remaining:.0f}s remaining)")
                
        cap.release()

        # Final timing
        total_time = time.time() - start_time
        print(f"\nCompleted in {total_time:.1f}s ({processed_count / total_time:.1f} fps)")

        # Save all frames
        final_output = {"frames": frames_data}
        with open(args.output, 'w') as f:
            json.dump(final_output, f, indent=2)

        print(f"Saved video motion data to {args.output} ({len(frames_data)} frames)")

    else:
        # Image Processing (Original Logic)
        print(f"Processing image {args.image}...")
        outputs = estimator.process_one_image(args.image)
    
        if not outputs:
            print("No person detected.")
            return
    
        # Process the first detection
        out_data = outputs[0]
        serializable_data = {k: to_list(v) for k, v in out_data.items()}
    
        with open(args.output, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
        print(f"Saved SAM3D data to {args.output}")

if __name__ == "__main__":
    main()
