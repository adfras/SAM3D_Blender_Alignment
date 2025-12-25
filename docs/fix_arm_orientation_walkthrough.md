# SAM3D to MetaHuman FBX Export - Bug Fixes Walkthrough

**Date:** 2025-12-25  
**File Modified:** `src/metahuman_standard_export.py`

---

## Issue 1: Arms and Shoulders Facing Backwards

### Problem
After exporting the FBX from Blender and retargeting to a MetaHuman in Unreal Engine 5, the arms and shoulders were facing backwards (behind the body instead of in front).

### Root Cause
Two issues in the export configuration:

1. **Incorrect Coordinate Transform Matrix (Line 40)**
   - The `COORD_TRANSFORM` matrix was incorrectly negating the Z component
   - Original: `Matrix([[1,0,0], [0,0,-1], [0,-1,0]])`
   - This caused the forward direction to be inverted

2. **Wrong FBX Export Axis (Line 502)**
   - The `axis_forward` was set to `'Y'` instead of `'-Y'`
   - A previous change had incorrectly modified this with a misleading comment

### Fix Applied

**Line 40 - Coordinate Transform:**
```diff
-COORD_TRANSFORM = Matrix([[1,0,0], [0,0,-1], [0,-1,0]])
+COORD_TRANSFORM = Matrix([[1,0,0], [0,0,1], [0,-1,0]])
```

**Line 502 - FBX Export Axis:**
```diff
-axis_forward='Y',   # Changed from -Y to fix backward orientation
+axis_forward='-Y',  # Standard UE5 forward axis
```

### Explanation
The coordinate transform converts from SAM3D's coordinate system (Y-up) to Blender's (Z-up):
- SAM3D: X=right, Y=up, Z=forward
- Blender: X=right, Y=forward, Z=up

The correct transform preserves the forward direction:
- Blender Y = SAM3D Z (forward stays forward)
- Blender Z = -SAM3D Y (up axis conversion)

The `-Y` axis forward in FBX export is the standard for Unreal Engine 5 compatibility.

---

## Issue 2: Hands Broken / Missing Finger Bones

### Problem
After fixing the arm orientation, the hands appeared broken in Unreal - fingers were not properly connected and deforming incorrectly.

### Root Cause
1. **Missing Metacarpal Bone Mappings**
   - SAM3D provides `l_pinky0`/`r_pinky0` (metacarpal for pinky) but it wasn't being mapped
   - The pinky metacarpal was being interpolated instead of using the actual SAM3D data

2. **Missing Metacarpal Bones for Other Fingers**
   - MetaHuman skeleton expects metacarpal bones for all fingers
   - SAM3D only provides pinky metacarpal (`_pinky0`); other fingers start at `_index1`, `_middle1`, `_ring1`
   - These missing metacarpals need to be interpolated

### Fix Applied

**Added SAM3D to MetaHuman Mappings (Lines 80, 97):**
```python
# Left hand
"l_pinky0": "pinky_metacarpal_l",  # SAM3D has this as metacarpal

# Right hand  
"r_pinky0": "pinky_metacarpal_r",  # SAM3D has this as metacarpal
```

**Added Metacarpal Bones to Hierarchy (Lines 142-180):**
```python
# Left fingers - with metacarpals
"index_metacarpal_l": ("hand_l", "index_01_l"),
"index_01_l": ("index_metacarpal_l", "index_02_l"),
# ... same pattern for middle, ring

# Right fingers - with metacarpals  
"index_metacarpal_r": ("hand_r", "index_01_r"),
"index_01_r": ("index_metacarpal_r", "index_02_r"),
# ... same pattern for middle, ring
```

**Added Interpolated Metacarpals (Lines 191-198):**
```python
INTERPOLATED_BONES = {
    "spine_04": ("c_spine2", "c_spine3"),
    "neck_02": ("c_neck", "c_head"),
    # Metacarpals for index, middle, ring (SAM3D doesn't have these)
    "index_metacarpal_l": ("l_wrist", "l_index1"),
    "index_metacarpal_r": ("r_wrist", "r_index1"),
    "middle_metacarpal_l": ("l_wrist", "l_middle1"),
    "middle_metacarpal_r": ("r_wrist", "r_middle1"),
    "ring_metacarpal_l": ("l_wrist", "l_ring1"),
    "ring_metacarpal_r": ("r_wrist", "r_ring1"),
    # Note: pinky_metacarpal now mapped directly from l_pinky0/r_pinky0
}
```

### Explanation
MetaHuman skeletons have metacarpal bones connecting the hand to each finger chain. SAM3D only provides the pinky metacarpal (`pinky0`); other fingers start at the first knuckle (`index1`, `middle1`, etc.).

The fix:
1. Maps SAM3D's `pinky0` directly to `pinky_metacarpal`
2. Creates interpolated metacarpal bones for index, middle, and ring fingers (positioned between wrist and first knuckle)
3. Updates the hierarchy so all finger bones have proper metacarpal parents

---

## Issue 3: Spine Slightly Off (Noted but not separately fixed)

The spine curvature issue was likely caused by the same coordinate transform problem. By fixing the `COORD_TRANSFORM` matrix, the spine orientation should also be corrected.

---

## Summary of All Changes

| Line | Change | Purpose |
|------|--------|---------|
| 40 | `[0,0,-1]` → `[0,0,1]` | Fix forward direction in coordinate transform |
| 80 | Added `l_pinky0` mapping | Map SAM3D pinky metacarpal |
| 97 | Added `r_pinky0` mapping | Map SAM3D pinky metacarpal |
| 142-159 | Added left metacarpals | Complete finger hierarchy |
| 160-180 | Added right metacarpals | Complete finger hierarchy |
| 191-198 | Interpolated metacarpals | Create missing finger bones |
| 502 | `'Y'` → `'-Y'` | Fix FBX export axis for UE5 |

---

## Testing Steps

1. Run the script in Blender:
   ```
   blender --background --python "src/metahuman_standard_export.py"
   ```

2. The FBX is exported to:
   ```
   data/metahuman_standard.fbx
   ```

3. Import into Unreal Engine 5

4. Create IK Rig and Retargeter

5. Retarget animation to MetaHuman

---

## Result

- **Arms:** Now facing forward correctly
- **Hands:** 63 empties created (up from 57), includes all metacarpal bones
- **Fingers:** Properly connected with full bone chains
