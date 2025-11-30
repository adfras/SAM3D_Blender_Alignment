import argparse
import os
import cv2
import numpy as np
import torch
import json
import sys

# Add sam-3d-body to path
sys.path.append(os.path.join(os.path.dirname(__file__), "sam-3d-body"))

try:
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
except ImportError:
    print("Could not import sam_3d_body. Make sure it is installed or in the python path.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="qYwLO.png")
    parser.add_argument("--checkpoint", help="Path to model checkpoint")
    parser.add_argument("--repo_id", default="facebook/sam-3d-body-dinov3", help="Hugging Face Repo ID")
    parser.add_argument("--output", default="sam3d_data.json")
    parser.add_argument("--mhr_path", default="")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

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

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None, # Assuming single person or center crop
        human_segmentor=None,
        fov_estimator=None,
    )

    # Load Image
    if not os.path.exists(args.image):
        print(f"Image {args.image} not found.")
        return
    
    print(f"Processing {args.image}...")
    # process_one_image handles loading if string is passed
    outputs = estimator.process_one_image(args.image)

    if not outputs:
        print("No person detected.")
        return

    # Save output
    # Convert numpy arrays to lists for JSON serialization
    def to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    # Process the first detection
    out_data = outputs[0]
    serializable_data = {k: to_list(v) for k, v in out_data.items()}

    with open(args.output, 'w') as f:
        json.dump(serializable_data, f, indent=2)

    print(f"Saved SAM3D data to {args.output}")

if __name__ == "__main__":
    main()
