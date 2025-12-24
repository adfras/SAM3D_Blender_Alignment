"""
Temporal Smoothing for SAM3D Motion Data
=========================================

Preprocesses video_motion_armature.json to reduce high-frequency jitter
in joint positions and rotations.

Smoothing Methods:
    - butterworth: Low-pass Butterworth filter (default, cutoff ~3Hz)
    - moving_avg: Simple moving average 
    - kalman: Kalman filter (position + velocity state)

Rotation Handling:
    - Converts 3x3 matrices to quaternions
    - Ensures quaternion continuity (sign flips)
    - Applies weighted SLERP chain for smoothing

Usage:
    python src/smooth_motion_data.py --input data/video_motion_armature.json --output data/video_motion_armature_smooth.json

Arguments:
    --input         Input JSON from run_sam3d_inference.py
    --output        Output smoothed JSON (default: adds _smooth suffix)
    --filter        Filter type: butterworth, moving_avg, kalman (default: butterworth)
    --cutoff        Cutoff frequency in Hz for butterworth (default: 3.0)
    --window        Window size for moving_avg (default: 5)
    --normalize-lengths  Enforce bone lengths from first frame
"""

import argparse
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

# Optional scipy import for Butterworth filter
try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Butterworth filter unavailable, using moving_avg.")


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m = np.array(matrix)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def ensure_quaternion_continuity(quaternions: np.ndarray) -> np.ndarray:
    """Flip quaternion signs to ensure continuous path (avoid antipodal jumps)."""
    result = quaternions.copy()
    for i in range(1, len(result)):
        if np.dot(result[i-1], result[i]) < 0:
            result[i] = -result[i]
    return result


def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    
    if dot > 0.9995:
        # Linear interpolation for very close quaternions
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q_perp = q2 - q1 * dot
    q_perp = q_perp / np.linalg.norm(q_perp)
    
    return q1 * np.cos(theta) + q_perp * np.sin(theta)


def smooth_quaternions_slerp(quaternions: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth quaternion sequence using weighted SLERP averaging."""
    n = len(quaternions)
    if n <= window:
        return quaternions.copy()
    
    # Ensure continuity first
    quats = ensure_quaternion_continuity(quaternions)
    result = np.zeros_like(quats)
    
    half_w = window // 2
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        
        # Weighted average (triangular weights centered on current frame)
        weights = []
        weighted_quats = []
        for j in range(start, end):
            w = 1.0 - abs(j - i) / (half_w + 1)
            weights.append(w)
            weighted_quats.append(quats[j])
        
        # Iterative SLERP to get weighted average
        avg_quat = weighted_quats[0]
        total_weight = weights[0]
        
        for j in range(1, len(weighted_quats)):
            t = weights[j] / (total_weight + weights[j])
            avg_quat = slerp(avg_quat, weighted_quats[j], t)
            total_weight += weights[j]
        
        result[i] = avg_quat / np.linalg.norm(avg_quat)
    
    return result


def butterworth_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    """Apply Butterworth low-pass filter to each column of data."""
    if not HAS_SCIPY:
        return moving_average_filter(data, window=5)
    
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    
    # Clamp to valid range
    normal_cutoff = np.clip(normal_cutoff, 0.01, 0.99)
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter to each dimension
    result = np.zeros_like(data)
    for dim in range(data.shape[1]):
        # Pad to avoid edge effects
        padlen = min(3 * max(len(a), len(b)), len(data) - 1)
        if padlen > 0 and len(data) > padlen:
            result[:, dim] = filtfilt(b, a, data[:, dim], padlen=padlen)
        else:
            result[:, dim] = data[:, dim]
    
    return result


def moving_average_filter(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average filter to each column of data."""
    n = len(data)
    if n <= window:
        return data.copy()
    
    result = np.zeros_like(data)
    half_w = window // 2
    
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        result[i] = np.mean(data[start:end], axis=0)
    
    return result


def kalman_filter_1d(data: np.ndarray, process_noise: float = 0.01, measurement_noise: float = 0.1) -> np.ndarray:
    """Simple 1D Kalman filter for position data."""
    n = len(data)
    result = np.zeros_like(data)
    
    for dim in range(data.shape[1]):
        # State: [position, velocity]
        x = np.array([data[0, dim], 0.0])
        P = np.eye(2)
        
        # State transition (constant velocity model)
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.eye(2) * process_noise
        R = np.array([[measurement_noise]])
        
        for i in range(n):
            # Predict
            x = F @ x
            P = F @ P @ F.T + Q
            
            # Update
            z = np.array([data[i, dim]])
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P
            
            result[i, dim] = x[0]
    
    return result


def smooth_positions(frames_data: List[Dict], filter_type: str, cutoff: float, window: int, fps: float) -> List[Dict]:
    """Smooth joint position trajectories across all frames."""
    n_frames = len(frames_data)
    if n_frames < 2:
        return frames_data
    
    # Extract position data for all joints
    # Use joints_mhr if available, otherwise joints3d
    sample_joints = frames_data[0].get('joints_mhr', frames_data[0].get('joints3d', []))
    n_joints = len(sample_joints)
    
    print(f"Smoothing {n_joints} joints across {n_frames} frames...")
    
    # Build position array: (n_frames, n_joints, 3)
    positions = np.zeros((n_frames, n_joints, 3))
    for i, frame in enumerate(frames_data):
        joints = frame.get('joints_mhr', frame.get('joints3d', []))
        for j, joint in enumerate(joints[:n_joints]):
            positions[i, j] = joint
    
    # Smooth each joint independently
    smoothed = np.zeros_like(positions)
    for j in range(n_joints):
        joint_traj = positions[:, j, :]  # (n_frames, 3)
        
        if filter_type == 'butterworth':
            smoothed[:, j, :] = butterworth_filter(joint_traj, cutoff, fps)
        elif filter_type == 'kalman':
            smoothed[:, j, :] = kalman_filter_1d(joint_traj)
        else:  # moving_avg
            smoothed[:, j, :] = moving_average_filter(joint_traj, window)
    
    # Write back to frames
    for i, frame in enumerate(frames_data):
        key = 'joints_mhr' if 'joints_mhr' in frame else 'joints3d'
        frame[key] = smoothed[i].tolist()
    
    return frames_data


def smooth_rotations(frames_data: List[Dict], window: int) -> List[Dict]:
    """Smooth rotation matrices using quaternion SLERP."""
    n_frames = len(frames_data)
    if n_frames < 2:
        return frames_data
    
    # Check if rotations exist
    if 'joint_rotations' not in frames_data[0]:
        print("No joint_rotations found, skipping rotation smoothing.")
        return frames_data
    
    sample_rots = frames_data[0]['joint_rotations']
    n_joints = len(sample_rots)
    
    print(f"Smoothing {n_joints} rotations across {n_frames} frames...")
    
    # Convert all rotations to quaternions
    quaternions = np.zeros((n_frames, n_joints, 4))
    for i, frame in enumerate(frames_data):
        rots = frame['joint_rotations']
        for j, rot in enumerate(rots[:n_joints]):
            quaternions[i, j] = quaternion_from_matrix(rot)
    
    # Smooth each joint's rotation independently
    smoothed_quats = np.zeros_like(quaternions)
    for j in range(n_joints):
        joint_quats = quaternions[:, j, :]  # (n_frames, 4)
        smoothed_quats[:, j, :] = smooth_quaternions_slerp(joint_quats, window)
    
    # Convert back to rotation matrices
    for i, frame in enumerate(frames_data):
        new_rots = []
        for j in range(n_joints):
            mat = matrix_from_quaternion(smoothed_quats[i, j])
            new_rots.append(mat.tolist())
        frame['joint_rotations'] = new_rots
    
    return frames_data


def normalize_bone_lengths(frames_data: List[Dict], hierarchy: Optional[Dict] = None) -> List[Dict]:
    """Enforce bone lengths from first frame throughout the sequence."""
    if len(frames_data) < 2:
        return frames_data
    
    # Get reference positions from first frame
    ref_joints = frames_data[0].get('joints_mhr', frames_data[0].get('joints3d', []))
    ref_positions = np.array(ref_joints)
    n_joints = len(ref_positions)
    
    # If no hierarchy provided, use simple parent-child assumption
    # (each joint's parent is the previous joint, except root)
    parents = list(range(-1, n_joints - 1))
    if hierarchy and 'parents' in hierarchy:
        parents = hierarchy['parents']
    
    # Compute reference bone lengths
    ref_lengths = np.zeros(n_joints)
    for j in range(n_joints):
        parent_idx = parents[j]
        if parent_idx >= 0 and parent_idx < n_joints:
            ref_lengths[j] = np.linalg.norm(ref_positions[j] - ref_positions[parent_idx])
    
    print(f"Normalizing bone lengths for {n_joints} joints...")
    
    # Normalize each frame
    for frame in frames_data[1:]:  # Skip first frame (reference)
        key = 'joints_mhr' if 'joints_mhr' in frame else 'joints3d'
        joints = np.array(frame[key])
        
        # Process in parent order (root first)
        for j in range(n_joints):
            parent_idx = parents[j]
            if parent_idx >= 0 and parent_idx < n_joints and ref_lengths[j] > 0.001:
                # Get direction from parent to child
                direction = joints[j] - joints[parent_idx]
                current_length = np.linalg.norm(direction)
                
                if current_length > 0.001:
                    # Rescale to reference length
                    direction = direction / current_length * ref_lengths[j]
                    joints[j] = joints[parent_idx] + direction
        
        frame[key] = joints.tolist()
    
    return frames_data


def main():
    parser = argparse.ArgumentParser(description="Smooth SAM3D motion data")
    parser.add_argument("--input", default="data/video_motion_armature.json",
                        help="Input JSON file")
    parser.add_argument("--output", default=None,
                        help="Output JSON file (default: input_smooth.json)")
    parser.add_argument("--filter", choices=["butterworth", "moving_avg", "kalman"],
                        default="butterworth", help="Filter type for positions")
    parser.add_argument("--cutoff", type=float, default=3.0,
                        help="Cutoff frequency (Hz) for butterworth filter")
    parser.add_argument("--window", type=int, default=5,
                        help="Window size for moving average")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video frame rate")
    parser.add_argument("--normalize-lengths", action="store_true",
                        help="Enforce bone lengths from first frame")
    parser.add_argument("--hierarchy", default=None,
                        help="Hierarchy JSON file (for bone normalization)")
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(base_dir, input_path)
    
    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_smooth{ext}"
    elif not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
    
    print("=" * 60)
    print("SAM3D Motion Data Smoother")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Filter: {args.filter} (cutoff={args.cutoff}Hz, window={args.window})")
    print()
    
    # Load data
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    frames_data = data.get('frames', [data])
    print(f"Loaded {len(frames_data)} frames")
    
    # Load hierarchy if specified
    hierarchy = None
    if args.hierarchy:
        hier_path = args.hierarchy
        if not os.path.isabs(hier_path):
            hier_path = os.path.join(base_dir, hier_path)
        if os.path.exists(hier_path):
            with open(hier_path, 'r') as f:
                hierarchy = json.load(f)
    
    # Apply smoothing
    print("\n[1/3] Smoothing positions...")
    frames_data = smooth_positions(frames_data, args.filter, args.cutoff, args.window, args.fps)
    
    print("\n[2/3] Smoothing rotations...")
    frames_data = smooth_rotations(frames_data, args.window)
    
    if args.normalize_lengths:
        print("\n[3/3] Normalizing bone lengths...")
        frames_data = normalize_bone_lengths(frames_data, hierarchy)
    else:
        print("\n[3/3] Skipping bone length normalization (use --normalize-lengths to enable)")
    
    # Save output
    data['frames'] = frames_data
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print()
    print("=" * 60)
    print(f"Saved smoothed data to: {output_path}")
    print(f"Frames: {len(frames_data)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
