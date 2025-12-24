import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import uniform_filter1d
import argparse
import os

# --- Configuration ---
MOTION_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "video_motion.json")
DEFAULT_SMOOTHING_WINDOW = 5  # frames for temporal smoothing

# MHR70 joint names (SAM3D output format)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_hip, 10: right_hip, 11: left_knee, 12: right_knee
# 13: left_ankle, 14: right_ankle
# 15-20: foot keypoints, 21-61: hand keypoints, 62: left_wrist, 41: right_wrist
# 63-66: elbow details, 67-68: acromion, 69: neck

# MHR70 skeleton connections (from skeleton_info in mhr70.py)
MHR70_BONES = [
    # Legs
    (13, 11),  # left_ankle -> left_knee
    (11, 9),   # left_knee -> left_hip
    (14, 12),  # right_ankle -> right_knee
    (12, 10),  # right_knee -> right_hip
    (9, 10),   # left_hip -> right_hip
    # Torso
    (5, 9),    # left_shoulder -> left_hip
    (6, 10),   # right_shoulder -> right_hip
    (5, 6),    # left_shoulder -> right_shoulder
    # Neck and Head
    (5, 69),   # left_shoulder -> neck
    (6, 69),   # right_shoulder -> neck
    (69, 0),   # neck -> nose (head connection)
    # Arms
    (5, 7),    # left_shoulder -> left_elbow
    (6, 8),    # right_shoulder -> right_elbow
    (7, 62),   # left_elbow -> left_wrist
    (8, 41),   # right_elbow -> right_wrist
    # Head details
    (1, 2),    # left_eye -> right_eye
    (0, 1),    # nose -> left_eye
    (0, 2),    # nose -> right_eye
    (1, 3),    # left_eye -> left_ear
    (2, 4),    # right_eye -> right_ear
    # Feet
    (13, 15),  # left_ankle -> left_big_toe
    (13, 17),  # left_ankle -> left_heel
    (14, 18),  # right_ankle -> right_big_toe
    (14, 20),  # right_ankle -> right_heel
]

def load_motion_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def apply_temporal_smoothing(all_joints, window_size=DEFAULT_SMOOTHING_WINDOW):
    """Apply temporal smoothing to reduce jitter between frames.

    Uses a uniform (moving average) filter along the time axis.

    Args:
        all_joints: numpy array of shape (num_frames, num_joints, 3)
        window_size: number of frames for moving average (odd recommended)

    Returns:
        Smoothed joints array of same shape
    """
    if window_size <= 1:
        return all_joints

    smoothed = np.zeros_like(all_joints)
    # Apply filter along time axis (axis=0) for each joint coordinate
    for joint_idx in range(all_joints.shape[1]):
        for coord in range(3):
            smoothed[:, joint_idx, coord] = uniform_filter1d(
                all_joints[:, joint_idx, coord],
                size=window_size,
                mode='nearest'
            )
    return smoothed


# Leg joint indices in MHR70
LEG_JOINTS = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # hips, knees, ankles, feet


def apply_leg_smoothing(all_joints, window_size=9):
    """Apply extra smoothing specifically to leg joints to reduce glitches.

    Leg joints are prone to glitches when legs cross or come close together
    due to depth ambiguity in monocular 3D pose estimation.

    Args:
        all_joints: numpy array of shape (num_frames, num_joints, 3)
        window_size: smoothing window for legs (larger = smoother)

    Returns:
        Joints array with extra-smoothed legs
    """
    smoothed = all_joints.copy()

    for joint_idx in LEG_JOINTS:
        if joint_idx >= all_joints.shape[1]:
            continue
        for coord in range(3):
            smoothed[:, joint_idx, coord] = uniform_filter1d(
                all_joints[:, joint_idx, coord],
                size=window_size,
                mode='nearest'
            )
    return smoothed


def fix_leg_crossover_glitches(all_joints, velocity_threshold=0.15):
    """Fix sudden leg position jumps that occur during crossover.

    When legs cross, pose estimation sometimes swaps left/right or makes
    sudden jumps. This detects and interpolates over those glitches.

    Args:
        all_joints: numpy array of shape (num_frames, num_joints, 3)
        velocity_threshold: max allowed position change per frame (as fraction of leg length)

    Returns:
        Fixed joints array
    """
    fixed = all_joints.copy()
    num_frames = all_joints.shape[0]

    # Estimate typical leg length from first frame
    if all_joints.shape[1] > 13:
        hip_to_knee = np.linalg.norm(all_joints[0, 9] - all_joints[0, 11])  # left
        knee_to_ankle = np.linalg.norm(all_joints[0, 11] - all_joints[0, 13])
        leg_length = hip_to_knee + knee_to_ankle
    else:
        leg_length = 1.0

    max_velocity = velocity_threshold * leg_length

    for joint_idx in LEG_JOINTS:
        if joint_idx >= all_joints.shape[1]:
            continue

        for frame in range(1, num_frames):
            velocity = np.linalg.norm(fixed[frame, joint_idx] - fixed[frame-1, joint_idx])

            if velocity > max_velocity:
                # Sudden jump detected - interpolate from previous position
                # Look ahead to find next stable frame
                stable_frame = frame + 1
                while stable_frame < num_frames - 1:
                    future_vel = np.linalg.norm(
                        all_joints[stable_frame + 1, joint_idx] - all_joints[stable_frame, joint_idx]
                    )
                    if future_vel < max_velocity:
                        break
                    stable_frame += 1

                # Interpolate between last good frame and next stable frame
                if stable_frame < num_frames:
                    t = 1.0 / max(1, stable_frame - frame + 1)
                    fixed[frame, joint_idx] = (
                        fixed[frame - 1, joint_idx] * (1 - t) +
                        all_joints[stable_frame, joint_idx] * t
                    )
                else:
                    # No stable frame found, just use previous
                    fixed[frame, joint_idx] = fixed[frame - 1, joint_idx]

    return fixed


def normalize_skeleton_height(joints, target_height=1.8):
    """Scale skeleton to a consistent height.

    Args:
        joints: (num_joints, 3) array
        target_height: desired height in arbitrary units (default 1.8)

    Returns:
        Scaled joints array and the scale factor used
    """
    # Calculate current height from head to feet
    # Use nose (0) for top and average of ankles (13, 14) for bottom
    if len(joints) > 14:
        top_z = joints[0, 2]  # nose Z
        bottom_z = (joints[13, 2] + joints[14, 2]) / 2.0  # ankles Z
        current_height = abs(top_z - bottom_z)
    else:
        current_height = np.max(joints[:, 2]) - np.min(joints[:, 2])

    if current_height < 0.01:  # avoid division by zero
        return joints, 1.0

    scale = target_height / current_height
    return joints * scale, scale


def align_joints(joints, ground_align=True):
    """Center and rotate joints for visualization (Z-up in matplotlib).

    Args:
        joints: (num_joints, 3) array of joint positions
        ground_align: if True, shift skeleton so feet touch z=0 plane

    Returns:
        Aligned joints array
    """
    joints = np.array(joints)

    # Center on hip midpoint (joints 9=left_hip, 10=right_hip) for stability
    if len(joints) > 10:
        root = (joints[9] + joints[10]) / 2.0
    else:
        root = joints[0]
    centered = joints - root

    # SAM3D coords: X-right, Y-down, Z-forward (camera coords)
    # Matplotlib 3D: X-right, Y-forward, Z-up
    #
    # To make the skeleton face the viewer (front view) without moonwalking:
    # - Keep X as-is (no flip) so left/right motion is preserved
    # - Negate Z for depth (so person faces camera, not away)
    # - Negate Y for height (Y-down becomes Z-up)
    rotated = np.zeros_like(centered)
    rotated[:, 0] = centered[:, 0]      # X stays (left-right)
    rotated[:, 1] = -centered[:, 2]     # Y = -Z (flip depth so facing viewer)
    rotated[:, 2] = -centered[:, 1]     # Z = -Y (up)

    # Ground plane alignment: shift so lowest point is at z=0
    if ground_align:
        min_z = np.min(rotated[:, 2])
        rotated[:, 2] -= min_z

    return rotated

# MHR70 hand skeleton connections (joints 21-61 are hand keypoints)
# Left hand: 21-40, Right hand: 41-61 (note: 41 is also right_wrist)
# Hand layout: wrist -> each finger base -> finger joints
MHR70_HAND_BONES = [
    # Left hand (wrist at joint 62)
    (62, 21), (21, 22), (22, 23), (23, 24),  # left thumb
    (62, 25), (25, 26), (26, 27), (27, 28),  # left index
    (62, 29), (29, 30), (30, 31), (31, 32),  # left middle
    (62, 33), (33, 34), (34, 35), (35, 36),  # left ring
    (62, 37), (37, 38), (38, 39), (39, 40),  # left pinky
    # Right hand (wrist at joint 41)
    (41, 42), (42, 43), (43, 44), (44, 45),  # right thumb
    (41, 46), (46, 47), (47, 48), (48, 49),  # right index
    (41, 50), (50, 51), (51, 52), (52, 53),  # right middle
    (41, 54), (54, 55), (55, 56), (56, 57),  # right ring
    (41, 58), (58, 59), (59, 60), (60, 61),  # right pinky
]


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM3D video inference results")
    parser.add_argument("--motion", default=MOTION_FILE, help="Path to motion JSON")
    parser.add_argument("--output", default="video_inference.gif", help="Output GIF path")
    parser.add_argument("--video", help="Source video (unused, for compatibility)")
    parser.add_argument("--smoothing", type=int, default=DEFAULT_SMOOTHING_WINDOW,
                        help=f"Temporal smoothing window size (0 to disable, default: {DEFAULT_SMOOTHING_WINDOW})")
    parser.add_argument("--fix-legs", action="store_true",
                        help="Apply extra leg smoothing and glitch fixing for crossover artifacts")
    parser.add_argument("--leg-smoothing", type=int, default=9,
                        help="Extra smoothing window for legs (default: 9, used with --fix-legs)")
    parser.add_argument("--normalize-height", action="store_true",
                        help="Normalize skeleton to consistent height")
    parser.add_argument("--show-hands", action="store_true",
                        help="Show hand skeleton (finger joints)")
    parser.add_argument("--show-confidence", action="store_true",
                        help="Color-code joints by detection confidence (if available)")
    parser.add_argument("--camera-angle", type=str, default="front",
                        choices=["front", "side", "top", "rotate"],
                        help="Camera view angle (default: front)")
    parser.add_argument("--fps", type=int, default=24, help="Output GIF framerate")
    args = parser.parse_args()

    print("Starting Video Inference Visualization...")
    print(f"  Smoothing window: {args.smoothing}")
    print(f"  Fix legs: {args.fix_legs}" + (f" (window={args.leg_smoothing})" if args.fix_legs else ""))
    print(f"  Camera angle: {args.camera_angle}")
    print(f"  Show hands: {args.show_hands}")

    if not os.path.exists(args.motion):
        print(f"Error: Motion file not found at {args.motion}")
        return

    data = load_motion_data(args.motion)
    if "frames" not in data:
        print("Error: JSON does not contain 'frames' key.")
        return

    frames = data["frames"]
    num_frames = len(frames)
    num_joints = len(frames[0]["joints3d"])
    print(f"Detected {num_joints} joints per frame (MHR70 format), {num_frames} frames")

    # Extract all joints into numpy array for smoothing
    all_joints_raw = np.array([f["joints3d"] for f in frames])

    # Apply temporal smoothing before alignment
    if args.smoothing > 1:
        print(f"Applying temporal smoothing (window={args.smoothing})...")
        all_joints_raw = apply_temporal_smoothing(all_joints_raw, args.smoothing)

    # Apply leg-specific fixes for crossover glitches
    if args.fix_legs:
        print(f"Fixing leg crossover glitches...")
        all_joints_raw = fix_leg_crossover_glitches(all_joints_raw)
        print(f"Applying extra leg smoothing (window={args.leg_smoothing})...")
        all_joints_raw = apply_leg_smoothing(all_joints_raw, args.leg_smoothing)

    # Pre-compute aligned joints for all frames
    print("Aligning joints for all frames...")
    all_joints_aligned = np.array([align_joints(j) for j in all_joints_raw])

    # Optional height normalization
    if args.normalize_height:
        print("Normalizing skeleton height...")
        for i in range(len(all_joints_aligned)):
            all_joints_aligned[i], _ = normalize_skeleton_height(all_joints_aligned[i])

    # Check for confidence data
    has_confidence = "confidence" in frames[0] if frames else False
    if args.show_confidence and not has_confidence:
        print("Warning: No confidence data found in motion file, ignoring --show-confidence")
        args.show_confidence = False

    # Setup Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Select bone connections
    bones = list(MHR70_BONES)
    if args.show_hands:
        bones.extend(MHR70_HAND_BONES)

    # Initialize lines (Red Skeleton, thinner for hands)
    lines = []
    for c_idx, p_idx in MHR70_BONES:
        line, = ax.plot([], [], [], color='#ff0000', linewidth=2.5)
        lines.append((line, c_idx, p_idx))

    if args.show_hands:
        for c_idx, p_idx in MHR70_HAND_BONES:
            line, = ax.plot([], [], [], color='#ff8800', linewidth=1.5)  # Orange for hands
            lines.append((line, c_idx, p_idx))

    # Add joint markers for main body joints
    scatter = ax.scatter([], [], [], c='#ff4444', s=30)

    # Hand joint markers (smaller)
    hand_scatter = None
    if args.show_hands:
        hand_scatter = ax.scatter([], [], [], c='#ffaa44', s=15)

    # Compute bounds from first frame
    first_joints = all_joints_aligned[0]
    max_range = np.max(np.abs(first_joints[:15])) * 1.5  # Use main body for bounds

    # For normalized height, use fixed bounds
    if args.normalize_height:
        max_range = 1.5

    ax.set_xlim3d([-max_range, max_range])
    ax.set_ylim3d([-max_range, max_range])
    ax.set_zlim3d([0, max_range * 2])  # Z starts at 0 (ground plane)
    ax.set_xlabel('X')
    ax.set_ylabel('Y (depth)')
    ax.set_zlabel('Z (up)')
    ax.set_title("SAM3D Video Inference")

    # Set Camera View based on argument
    camera_views = {
        "front": (15, 90),
        "side": (15, 0),
        "top": (90, 90),
    }
    if args.camera_angle in camera_views:
        elev, azim = camera_views[args.camera_angle]
        ax.view_init(elev=elev, azim=azim)
    # "rotate" handled in update function

    def update(frame_idx):
        joints = all_joints_aligned[frame_idx]

        # Update bones
        for line, c_idx, p_idx in lines:
            if c_idx < len(joints) and p_idx < len(joints):
                c = joints[c_idx]
                p = joints[p_idx]
                line.set_data([c[0], p[0]], [c[1], p[1]])
                line.set_3d_properties([c[2], p[2]])

        # Update scatter - main body joints (0-14, plus neck 69)
        main_indices = list(range(15)) + [69] if num_joints > 69 else list(range(min(15, num_joints)))
        main_joints = joints[main_indices]

        if args.show_confidence and has_confidence:
            # Color by confidence (green = high, red = low)
            conf = np.array(frames[frame_idx]["confidence"])[main_indices]
            colors = plt.cm.RdYlGn(conf)
            scatter.set_facecolors(colors)

        scatter._offsets3d = (main_joints[:, 0], main_joints[:, 1], main_joints[:, 2])

        # Update hand joints if showing
        if args.show_hands and hand_scatter is not None:
            hand_indices = list(range(21, min(62, num_joints)))
            if hand_indices:
                hand_joints = joints[hand_indices]
                hand_scatter._offsets3d = (hand_joints[:, 0], hand_joints[:, 1], hand_joints[:, 2])

        # Rotating camera view
        if args.camera_angle == "rotate":
            ax.view_init(elev=15, azim=(90 + frame_idx * 2) % 360)

        artists = [l[0] for l in lines] + [scatter]
        if hand_scatter is not None:
            artists.append(hand_scatter)
        return artists

    print(f"Animating {num_frames} frames at {args.fps} fps...")
    interval = 1000 // args.fps  # milliseconds per frame
    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)

    print(f"Saving animation to {args.output}...")
    try:
        anim.save(args.output, writer='pillow', fps=args.fps)
        print(f"Saved {args.output}")
    except Exception as e:
        print(f"Could not save GIF: {e}")
        plt.show()


if __name__ == "__main__":
    main()
