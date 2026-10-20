import os
import sys

WORKSPACE = r"e:\git_projects\Projects\Visiona3D"
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from core.detection import Detection
from perception.vision import VisionSystem

def test_depth_cache_desync():
    print("=" * 65)
    print("REPRODUCTION: DEPTH-CACHE DESYNCHRONIZATION & SWAP")
    print("=" * 65)

    # Instantiate VisionSystem without loading large models
    # We will directly test the depth caching logic in perception/vision.py:114-129
    
    # Frame 1: Two detections
    # Det 1: Approaching Car (True depth: 1.5m)
    # Det 2: Distant Chair (True depth: 6.0m)
    car = Detection(label="car", direction="FRONT", confidence=0.9, box=(100, 100, 300, 300))
    chair = Detection(label="chair", direction="FRONT", confidence=0.8, box=(400, 100, 600, 300))
    
    detections_frame_1 = [car, chair]
    
    # Simulate the async depth thread completing and populating cached depths:
    # cached depths for [car, chair]
    cached_depths = [1.5, 6.0]
    
    print("\n[Frame 1 Setup]")
    print(f"  Object 0: {detections_frame_1[0].label} (Actual True Depth: 1.5m)")
    print(f"  Object 1: {detections_frame_1[1].label} (Actual True Depth: 6.0m)")
    print(f"  Async Depth Thread caches values: {cached_depths}")
    
    # Frame 2: Same 2 objects still present, but YOLO returns them in swapped order:
    # [chair, car] (e.g. due to detector sorting by confidence or camera motion)
    car_frame_2 = Detection(label="car", direction="FRONT", confidence=0.85, box=(100, 100, 300, 300))
    chair_frame_2 = Detection(label="chair", direction="FRONT", confidence=0.92, box=(400, 100, 600, 300))
    
    detections_frame_2 = [chair_frame_2, car_frame_2] # Swapped list order
    
    print("\n[Frame 2 Detections (Swapped Detection Order)]")
    print(f"  Object 0 in list: {detections_frame_2[0].label} (Ground truth: ~6.0m)")
    print(f"  Object 1 in list: {detections_frame_2[1].label} (Ground truth: ~1.5m)")
    
    # Now simulate vision.py lines 114-129 logic:
    cached = list(cached_depths)
    if cached and len(cached) == len(detections_frame_2):
        depths = cached
    else:
        depths = [999.0, 999.0]
        
    for det, dm in zip(detections_frame_2, depths):
        det.distance_m = dm
        det.depth_mode = "monocular"
        
    print("\n[Depth Assignment Results in vision.py:126-128]")
    print(f"  Result for {detections_frame_2[0].label}: assigned distance = {detections_frame_2[0].distance_m}m")
    print(f"  Result for {detections_frame_2[1].label}:   assigned distance = {detections_frame_2[1].distance_m}m")
    
    print("\n" + "=" * 65)
    print("CONCLUSION:")
    print(f"  The distant CHAIR was assigned {chair_frame_2.distance_m}m (inherited the CAR's 1.5m depth).")
    print(f"  The approaching CAR was assigned {car_frame_2.distance_m}m (inherited the CHAIR's 6.0m depth).")
    print("Cause: perception/vision.py:126 pairs cached depths solely by list index:")
    print("       for det, dm in zip(detections, depths): det.distance_m = dm")
    print("There is zero spatial overlap check, IoU tracking, or centroid matching")
    print("between cached depth ROIs and new frame detections.")
    print("=" * 65)

if __name__ == "__main__":
    test_depth_cache_desync()
