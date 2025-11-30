# SAM3D to Blender Skeleton Alignment

This project provides tools to align a SAM3D-derived skeleton (from pose estimation) with a Blender MHR armature. It achieves a "perfect alignment" by reconstructing the SAM3D skeleton using the exact bone vectors and topology of the Blender armature, ensuring 1:1 compatibility for animation retargeting.

## Key Features

- **Hierarchical Reconstruction**: Rebuilds the SAM3D skeleton bone-by-bone using Blender's rest pose vectors.
- **Perfect Alignment**: Guarantees zero rotational or scaling errors between the source and target skeletons.
- **Visual Verification**: Includes tools to visualize the skeletons side-by-side or superimposed to verify alignment.

## Project Structure

- **`skeleton_core.py`**: The core library containing all alignment logic, including data loading, parsing, and the reconstruction algorithm.
- **`run_alignment.py`**: The main script to execute the alignment pipeline. It generates:
    - `aligned_skeleton_data.json`: The final aligned skeleton data.
    - `debug_joint_positions.csv`: Joint positions for verification.
    - `debug_bone_vectors.csv`: Bone vector metrics for verification.
- **`comparison_sidebyside.py`**: Visualizes the Blender (Green) and SAM3D (Red) skeletons side-by-side.
- **`visualize_superimposed.py`**: Visualizes the skeletons superimposed to demonstrate the perfect match.
- **`verify_data_sources.py`**: Verifies that the input data sources are distinct (proving the alignment is a reconstruction, not a copy).

## Data Files

- `sam3d_data.json`: Input pose estimation data.
- `blender_rest_pose.json`: Reference rest pose from Blender.
- `mhr_hierarchy.json`: Bone hierarchy definition.
- `qYwLO.png`: Reference image.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Alignment**:
    ```bash
    python run_alignment.py
    ```

3.  **Visualize Results**:
    ```bash
    python comparison_sidebyside.py
    python visualize_superimposed.py
    ```

## Archive

Legacy scripts and debug outputs are moved to the `archive/` directory (not included in the main repository).
