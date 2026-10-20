import os
import sys
import numpy as np

WORKSPACE = r"e:\git_projects\Projects\Visiona3D"
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from perception.mono_depth import MonoDepth, _REF_WIDTHS_M
from core.config import Config

class StubMonoDepth(MonoDepth):
    def __init__(self, stub_depth_map):
        super().__init__()
        self._stub_map = stub_depth_map
        self._model = "STUB_LOADED"
        
    def _run_midas(self, frame):
        return self._stub_map

def test_depth_scale():
    print("=" * 60)
    print("REPRODUCTION: MONO_DEPTH X2.0 SCALE ISSUE")
    print("=" * 60)
    
    # Frame resolution: 1280x720 (orig_w = 1280, focal_px = 1280 * 0.73 = 934.4)
    # real_dist = (0.50 * 934.4) / px_w
    frame_h, frame_w = 720, 1280
    focal_px = frame_w * 0.73  # 934.4 px
    real_w_m = _REF_WIDTHS_M["person"] # 0.50 m
    
    # Use real distances <= 3.5m so that 2x does not hit Config.MAX_DISTANCE_M (7.5m)
    test_cases = [
        {"name": "Distance 1 (~1.2m)", "target_dist": 1.2},
        {"name": "Distance 2 (~2.0m)", "target_dist": 2.0},
        {"name": "Distance 3 (~3.2m)", "target_dist": 3.2},
    ]
    
    for case in test_cases:
        target = case["target_dist"]
        px_w = int(round((real_w_m * focal_px) / target))
        expected_real_dist = round((real_w_m * focal_px) / px_w, 2)
        
        # Bounding box of width px_w centered in frame
        x1 = (frame_w - px_w) // 2
        x2 = x1 + px_w
        y1 = 200
        y2 = 600
        box = (x1, y1, x2, y2)
        
        d_val = 350.0
        depth_map = np.full((frame_h, frame_w), d_val, dtype=np.float32)
        
        mono = StubMonoDepth(depth_map)
        dummy_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        
        # 1. Run calibration on reference object
        mono._calibrate(depth_map, [box], ["person"], (frame_h, frame_w))
        calibrated_scale = mono.scale
        
        # 2. Run compute() on the exact same detection
        computed_depths = mono.compute(dummy_frame, [box], labels=["person"])
        returned_dist = computed_depths[0]
        
        ratio = returned_dist / expected_real_dist if expected_real_dist else 0
        
        print(f"\nTest: {case['name']}")
        print(f"  Pixel Width       : {px_w} px")
        print(f"  Real Distance (m) : {expected_real_dist} m")
        print(f"  Calibrated Scale  : {calibrated_scale:.2f} (real_dist {expected_real_dist} * d_ref {d_val})")
        print(f"  Returned Distance : {returned_dist} m")
        print(f"  Ratio (Ret / Real): {ratio:.2f}x")
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("The returned distance is consistently 2.0x the real distance across all un-clamped tests.")
    print("Cause: perception/mono_depth.py:95 hardcodes an arbitrary '* 2.0' multiplier:")
    print("       metric = round((self._scale / midas_val) * 2.0, 2)")
    print("Because _scale was already calibrated to real_dist * d_ref, multiplying by 2.0")
    print("doubles the estimated distance over real ground truth.")
    print("=" * 60)

if __name__ == "__main__":
    test_depth_scale()
