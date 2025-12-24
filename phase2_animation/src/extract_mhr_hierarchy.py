"""
Extract MHR skeleton hierarchy from SAM3D model.

This script extracts the 127-joint MHR skeleton structure (joint names and parent indices)
from the SAM3D model. This is required for armature creation in Blender.

Usage:
    python src/extract_mhr_hierarchy.py

Output:
    data/mhr_hierarchy.json - Contains 'joints' (list of 127 joint names) and 
                              'parents' (list of parent indices, -1 for root)
"""

import sys
import os
import json
import torch

# Add sam-3d-body to path
# When run from SAM3D2Blender/src/, this resolves to skeleton_alignment_work/sam-3d-body
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM3D_PATH = os.path.join(SCRIPT_DIR, "..", "..", "sam-3d-body")
if not os.path.exists(SAM3D_PATH):
    # Fallback: check if sam-3d-body is in the same parent as workspace root
    SAM3D_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "sam-3d-body")
if not os.path.exists(SAM3D_PATH):
    print(f"WARNING: sam-3d-body not found. Expected at: {SAM3D_PATH}")
sys.path.append(SAM3D_PATH)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mhr_hierarchy.json")


def derive_parents_from_names(joint_names):
    """
    Derive parent indices from MHR joint naming convention.
    
    MHR uses a hierarchical naming scheme:
    - root is the only joint with parent -1
    - l_/r_/c_ prefixes indicate side (left/right/center)
    - Numbers indicate hierarchy level (e.g., l_pinky0 -> l_pinky1 -> l_pinky2)
    - _twist/_proc suffixes indicate derived bones
    - Explicit hierarchy based on known skeleton structure
    """
    n = len(joint_names)
    parents = [-1] * n  # Default: no parent
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    
    # Known MHR skeleton hierarchy structure (child -> parent mappings)
    hierarchy_rules = {
        # Legs
        'l_upleg': 'root', 'r_upleg': 'root',
        'l_lowleg': 'l_upleg', 'r_lowleg': 'r_upleg',
        'l_foot': 'l_lowleg', 'r_foot': 'r_lowleg',
        'l_talocrural': 'l_foot', 'r_talocrural': 'r_foot',
        'l_subtalar': 'l_talocrural', 'r_subtalar': 'r_talocrural',
        'l_transversetarsal': 'l_subtalar', 'r_transversetarsal': 'r_subtalar',
        'l_ball': 'l_transversetarsal', 'r_ball': 'r_transversetarsal',
        
        # Spine
        'c_spine0': 'root',
        'c_spine1': 'c_spine0', 'c_spine2': 'c_spine1', 'c_spine3': 'c_spine2',
        
        # Arms (from spine3)
        'l_clavicle': 'c_spine3', 'r_clavicle': 'c_spine3',
        'l_uparm': 'l_clavicle', 'r_uparm': 'r_clavicle',
        'l_lowarm': 'l_uparm', 'r_lowarm': 'r_uparm',
        'l_wrist_twist': 'l_lowarm', 'r_wrist_twist': 'r_lowarm',
        'l_wrist': 'l_wrist_twist', 'r_wrist': 'r_wrist_twist',
        
        # Hands - pinky
        'l_pinky0': 'l_wrist', 'r_pinky0': 'r_wrist',
        'l_pinky1': 'l_pinky0', 'r_pinky1': 'r_pinky0',
        'l_pinky2': 'l_pinky1', 'r_pinky2': 'r_pinky1',
        'l_pinky3': 'l_pinky2', 'r_pinky3': 'r_pinky2',
        'l_pinky_null': 'l_pinky3', 'r_pinky_null': 'r_pinky3',
        
        # Hands - ring (no ring0)
        'l_ring1': 'l_wrist', 'r_ring1': 'r_wrist',
        'l_ring2': 'l_ring1', 'r_ring2': 'r_ring1',
        'l_ring3': 'l_ring2', 'r_ring3': 'r_ring2',
        'l_ring_null': 'l_ring3', 'r_ring_null': 'r_ring3',
        
        # Hands - middle
        'l_middle1': 'l_wrist', 'r_middle1': 'r_wrist',
        'l_middle2': 'l_middle1', 'r_middle2': 'r_middle1',
        'l_middle3': 'l_middle2', 'r_middle3': 'r_middle2',
        'l_middle_null': 'l_middle3', 'r_middle_null': 'r_middle3',
        
        # Hands - index
        'l_index1': 'l_wrist', 'r_index1': 'r_wrist',
        'l_index2': 'l_index1', 'r_index2': 'r_index1',
        'l_index3': 'l_index2', 'r_index3': 'r_index2',
        'l_index_null': 'l_index3', 'r_index_null': 'r_index3',
        
        # Hands - thumb
        'l_thumb0': 'l_wrist', 'r_thumb0': 'r_wrist',
        'l_thumb1': 'l_thumb0', 'r_thumb1': 'r_thumb0',
        'l_thumb2': 'l_thumb1', 'r_thumb2': 'r_thumb1',
        'l_thumb3': 'l_thumb2', 'r_thumb3': 'r_thumb2',
        'l_thumb_null': 'l_thumb3', 'r_thumb_null': 'r_thumb3',
        
        # Neck & Head
        'c_neck': 'c_spine3',
        'c_neck_twist1_proc': 'c_neck', 'c_neck_twist0_proc': 'c_neck',
        'c_head': 'c_neck',
        'c_jaw': 'c_head', 'c_teeth': 'c_jaw', 'c_jaw_null': 'c_jaw',
        'c_tongue0': 'c_head',
        'c_tongue1': 'c_tongue0', 'c_tongue2': 'c_tongue1',
        'c_tongue3': 'c_tongue2', 'c_tongue4': 'c_tongue3',
        'l_eye': 'c_head', 'r_eye': 'c_head',
        'l_eye_null': 'l_eye', 'r_eye_null': 'r_eye',
        'c_head_null': 'c_head',
    }
    
    # Apply explicit rules
    for child, parent in hierarchy_rules.items():
        if child in name_to_idx and parent in name_to_idx:
            parents[name_to_idx[child]] = name_to_idx[parent]
    
    # Handle twist/proc bones by finding their base bone
    for i, name in enumerate(joint_names):
        if parents[i] == -1 and name != 'root':
            # Try to find parent by removing _twist/_proc suffixes
            base = name
            for suffix in ['_twist1_proc', '_twist2_proc', '_twist3_proc', '_twist4_proc',
                          '_twist0_proc', '_twist1', '_twist2', '_twist', '_proc', '_null']:
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            
            # Find base bone or numbered predecessor
            if base in name_to_idx and base != name:
                parents[i] = name_to_idx[base]
            else:
                # Try numbered predecessor (e.g., l_upleg_twist1_proc -> l_upleg)
                for part in ['_twist', '_proc']:
                    if part in name:
                        base_name = name.split(part)[0]
                        if base_name in name_to_idx:
                            parents[i] = name_to_idx[base_name]
                            break
    
    return parents


def extract_hierarchy_from_model():
    """Extract joint names and parent indices from SAM3D model."""
    try:
        from sam_3d_body import load_sam_3d_body_hf
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SAM3D model on {device}...")
        
        model, model_cfg = load_sam_3d_body_hf("facebook/sam-3d-body-dinov3", device=device)
        
        # Search for joint names in the model
        joint_names = None
        parents = None
        
        # Check model_cfg first
        if model_cfg:
            for key in ['joint_names', 'joints', 'body_joint_names', 'mhr_joint_names']:
                if key in model_cfg:
                    joint_names = model_cfg[key]
                    print(f"Found joint names in model_cfg['{key}']")
                    break
            
            for key in ['parents', 'parent_indices', 'kinematic_tree']:
                if key in model_cfg:
                    parents = model_cfg[key]
                    print(f"Found parents in model_cfg['{key}']")
                    break
        
        # Search in model submodules more thoroughly
        def search_hierarchy(obj, path="", depth=0):
            nonlocal joint_names, parents
            
            # Avoid infinite recursion
            if depth > 5:
                return
            
            # Check for joint_names
            for attr in ['joint_names', 'body_joint_names', 'smpl_joint_names', 'mhr_joint_names', 
                         'JOINT_NAMES', 'BoneNames', 'bone_names']:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    if val is not None:
                        if isinstance(val, (list, tuple)) and len(val) > 50:
                            joint_names = list(val)
                            print(f"Found joint names in {path}.{attr} ({len(val)} joints)")
                        
            # Check for parents
            for attr in ['parents', 'kintree_parents', 'parent_indices', 'kintree_table',
                         'PARENTS', 'parent_ids', 'bone_parents']:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    if val is not None:
                        if hasattr(val, 'tolist'):
                            p = val.tolist()
                        elif hasattr(val, 'cpu'):
                            p = val.cpu().numpy().tolist()
                        elif isinstance(val, (list, tuple)):
                            p = list(val)
                        else:
                            continue
                        if len(p) > 50:
                            parents = p
                            print(f"Found parents in {path}.{attr}")
            
            # Deep recursive search through submodules
            sub_attrs = ['smpl', 'body_model', 'smpl_model', 'smpl_body', 'head_pose_hand', 
                        'mhr', 'character_torch', 'character', 'skeleton', 'rig']
            for sub_attr in sub_attrs:
                if hasattr(obj, sub_attr):
                    sub = getattr(obj, sub_attr)
                    if sub is not None and sub != obj:
                        search_hierarchy(sub, f"{path}.{sub_attr}", depth + 1)
            
            # Also check named children if it's a module
            if hasattr(obj, 'named_children'):
                try:
                    for name, child in obj.named_children():
                        if child is not None:
                            search_hierarchy(child, f"{path}.{name}", depth + 1)
                except:
                    pass
        
        search_hierarchy(model, "model")
        
        # Direct extraction at known path if not found
        if not joint_names or not parents:
            try:
                skeleton = model.head_pose_hand.mhr.character_torch.skeleton
                print(f"\n--- Exploring skeleton object at known path ---")
                
                # Print ALL non-private attributes to find parents
                attrs = [a for a in dir(skeleton) if not a.startswith('_')]
                print(f"ALL skeleton attributes ({len(attrs)}): {attrs}")
                
                # Also check the character_torch parent object
                char_torch = model.head_pose_hand.mhr.character_torch
                ct_attrs = [a for a in dir(char_torch) if not a.startswith('_')]
                print(f"\nALL character_torch attributes ({len(ct_attrs)}): {ct_attrs}")
                
                # Check for smpl on character_torch
                if hasattr(char_torch, 'smpl'):
                    smpl = char_torch.smpl
                    smpl_attrs = [a for a in dir(smpl) if not a.startswith('_')]
                    print(f"\nALL smpl attributes ({len(smpl_attrs)}): {smpl_attrs}")
                
                # Check the mhr object
                mhr = model.head_pose_hand.mhr
                mhr_attrs = [a for a in dir(mhr) if not a.startswith('_')]
                print(f"\nALL mhr attributes ({len(mhr_attrs)}): {mhr_attrs}")
                
                # Check for smpl on mhr
                if hasattr(mhr, 'smpl'):
                    smpl = mhr.smpl
                    smpl_attrs = [a for a in dir(smpl) if not a.startswith('_')]
                    print(f"\nALL mhr.smpl attributes ({len(smpl_attrs)}): {smpl_attrs}")
                    if hasattr(smpl, 'kintree_table'):
                        print(f"  FOUND kintree_table!")
                    if hasattr(smpl, 'parents'):
                        print(f"  FOUND parents!")
                
                # Try getting joint_names if still needed
                if not joint_names and hasattr(skeleton, 'joint_names'):
                    joint_names = list(skeleton.joint_names)
                    print(f"Direct extraction: {len(joint_names)} joint names")
                
                # Try common parent attribute names on skeleton
                for attr in ['parents', 'parent', 'parent_indices', 'kintree', 'kintree_parents',
                            'parent_ids', 'parent_index', 'topology', 'hierarchy']:
                    if hasattr(skeleton, attr):
                        val = getattr(skeleton, attr)
                        print(f"  skeleton.{attr} = {type(val).__name__}")
                        if val is not None:
                            # Try to convert to list
                            try:
                                if hasattr(val, 'tolist'):
                                    p = val.tolist()
                                elif hasattr(val, 'cpu'):
                                    p = val.cpu().numpy().tolist()
                                elif callable(val):
                                    continue
                                else:
                                    p = list(val) if hasattr(val, '__iter__') else None
                                if p and len(p) > 50:
                                    parents = p
                                    print(f"  -> Extracted {len(parents)} parent indices")
                                    break
                            except:
                                pass
            except Exception as e:
                print(f"Direct path exploration failed: {e}")
        
        # Try SMPL body model for parent indices (kintree_table)
        if not parents:
            try:
                # Search various possible paths for SMPL body model
                smpl_paths = [
                    'head_pose_hand.mhr.smpl',
                    'head_pose_hand.mhr.body_model', 
                    'head_pose_hand.smpl',
                    'head_pose.mhr.smpl',
                    'head_pose.mhr.body_model',
                    'head_pose_hand.mhr.character_torch.smpl',
                    'head_pose_hand.mhr.character_torch.body_model',
                ]
                
                for path in smpl_paths:
                    try:
                        obj = model
                        for attr in path.split('.'):
                            obj = getattr(obj, attr)
                        
                        print(f"\n--- Exploring SMPL at {path} ---")
                        attrs = [a for a in dir(obj) if not a.startswith('_') and 'parent' in a.lower() or 'kint' in a.lower()]
                        print(f"SMPL parent-related attrs: {attrs}")
                        
                        # Look for kintree_table (SMPL standard)
                        if hasattr(obj, 'kintree_table'):
                            kt = obj.kintree_table
                            print(f"  kintree_table shape: {kt.shape if hasattr(kt, 'shape') else type(kt)}")
                            if hasattr(kt, 'cpu'):
                                kt = kt.cpu().numpy()
                            # kintree_table[0] is parent indices, [1] is child indices 
                            if len(kt) >= 1:
                                parents = kt[0].tolist() if hasattr(kt[0], 'tolist') else list(kt[0])
                                print(f"  -> Extracted {len(parents)} parents from kintree_table[0]")
                                break
                        
                        if hasattr(obj, 'parents'):
                            p = obj.parents
                            if hasattr(p, 'cpu'):
                                p = p.cpu().numpy().tolist()
                            elif hasattr(p, 'tolist'):
                                p = p.tolist()
                            else:
                                p = list(p)
                            if len(p) > 20:
                                parents = p
                                print(f"  -> Extracted {len(parents)} parents")
                                break
                    except AttributeError:
                        continue
                    except Exception as e:
                        print(f"  Error at {path}: {e}")
            except Exception as e:
                print(f"SMPL body model exploration failed: {e}")
        
        # Fallback: derive parents from joint names if we have names but no parents
        if joint_names and not parents:
            print(f"\n--- Deriving parents from joint naming convention ---")
            parents = derive_parents_from_names(joint_names)
            orphans = sum(1 for i, p in enumerate(parents) if p == -1 and joint_names[i] != 'root')
            print(f"Derived {len(parents)} parent indices ({orphans} remaining orphans)")
        
        return joint_names, parents
        
    except Exception as e:
        print(f"Error extracting from SAM3D model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def save_hierarchy(joints, parents, output_path):
    """Save hierarchy to JSON file."""
    data = {
        "joints": joints,
        "parents": parents
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(joints)} joints to {output_path}")


def main():
    print("=" * 60)
    print("MHR Hierarchy Extraction")
    print("=" * 60)
    
    joints, parents = extract_hierarchy_from_model()
    
    if joints and parents:
        save_hierarchy(joints, parents, OUTPUT_PATH)
        print(f"\nSUCCESS! Hierarchy saved to: {OUTPUT_PATH}")
    else:
        print("\n" + "=" * 60)
        print("COULD NOT EXTRACT HIERARCHY AUTOMATICALLY")
        print("=" * 60)
        print("\nThe hierarchy data needs to be extracted manually from the SAM3D model.")
        print("Check the model's body_model or smpl submodule for joint_names and parents attributes.")


if __name__ == "__main__":
    main()
