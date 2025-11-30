# SAM3D to Blender Skeleton Alignment

This project provides tools to align a SAM3D-derived skeleton (from pose estimation) with a Blender MHR armature. It achieves a "perfect alignment" by reconstructing the SAM3D skeleton using the exact bone vectors and topology of the Blender armature, ensuring 1:1 compatibility for animation retargeting.

## Key Features

- **Hierarchical Reconstruction**: Rebuilds the SAM3D skeleton bone-by-bone using Blender's rest pose vectors.
- **Perfect Alignment**: Guarantees zero rotational or scaling errors between the source and target skeletons.
- **Visual Verification**: Includes tools to visualize the skeletons side-by-side or superimposed to verify alignment.

## Project Structure

- **`src/`**: Python source code.
    - **`skeleton_core.py`**: The core library containing all alignment logic.
    - **`run_alignment.py`**: The main script to execute the alignment pipeline.
    - **`comparison_sidebyside.py`**: Visualizes the skeletons side-by-side.
    - **`visualize_superimposed.py`**: Visualizes the skeletons superimposed.
- **`data/`**: Input data files (JSONs, images).
- **`docs/`**: Documentation and walkthroughs.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Alignment**:
    ```bash
    python src/run_alignment.py
    ```
    Outputs will be generated in the current directory.

3.  **Visualize Results**:
    ```bash
    python src/comparison_sidebyside.py
    python src/visualize_superimposed.py
    ```
