import os
import sys
import time
import traceback
import numpy as np

WORKSPACE = r"e:\git_projects\Projects\Visiona3D"
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

def run_smoke_test():
    print("=" * 70)
    print("VISIONA AI HEADLESS SMOKE TEST REPORT")
    print("=" * 70)
    
    first_failure = None
    
    # -------------------------------------------------------------------------
    # STAGE A: Import every module and report failures
    # -------------------------------------------------------------------------
    print("\n[STAGE A] IMPORT INTEGRITY CHECK")
    print("-" * 50)
    
    modules_to_test = [
        "core.config",
        "core.detection",
        "core.priority_queue",
        "core.logger",
        "core.memory",
        "core.recognition",
        "kinematics.kalman",
        "kinematics.speed",
        "kinematics.ttc",
        "kinematics.heatmap",
        "perception.egomotion",
        "perception.mono_depth",
        "perception.tracker",
        "perception.vision",
        "audio.alert",
        "audio.speech",
        "audio.voice_input",
        "services.vector_db",
        "agents.tools.vision_tools",
        "agents.orchestrator",
        "main",
    ]
    
    import_results = {}
    for mod_name in modules_to_test:
        try:
            __import__(mod_name)
            import_results[mod_name] = ("PASS", None)
            print(f"  [PASS] {mod_name}")
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            import_results[mod_name] = ("FAIL", err_msg)
            print(f"  [FAIL] {mod_name} -> {err_msg}")
            if first_failure is None:
                first_failure = f"Stage A: Import failure in {mod_name} ({err_msg})"

    # -------------------------------------------------------------------------
    # STAGE B: Synthetic Image Pipeline (Detection -> Tracking -> Kinematics)
    # -------------------------------------------------------------------------
    print("\n[STAGE B] SYNTHETIC FRAME PIPELINE (1280x720)")
    print("-" * 50)
    
    try:
        # Create synthetic 1280x720 BGR image (gray background with 2 rectangles)
        img = np.full((720, 1280, 3), 128, dtype=np.uint8)
        # Person rectangle (green)
        img[200:600, 300:500] = (0, 255, 0)
        # Car rectangle (blue)
        img[350:650, 700:1100] = (255, 0, 0)
        print("  Created synthetic 1280x720 image with simulated objects.")
        
        # 1. Detection Stage
        # Try real YOLO if possible, else FAKE
        from core.config import Config
        from core.detection import Detection
        
        yolo_model = None
        detections = []
        try:
            from ultralytics import YOLO
            if os.path.exists(Config.MODEL_PATH):
                print(f"  Loading real YOLO model from {Config.MODEL_PATH}...")
                yolo_model = YOLO(Config.MODEL_PATH)
                results = yolo_model(img, imgsz=Config.IMG_SIZE, verbose=False, device='cpu')
                for r in results:
                    for box in r.boxes:
                        conf = float(box.conf[0])
                        if conf >= Config.CONF_THRESHOLD:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = yolo_model.names[int(box.cls[0])]
                            detections.append(Detection(
                                label=label, direction="FRONT", confidence=conf, box=(x1, y1, x2, y2)
                            ))
                print(f"  [REAL YOLO] Detected {len(detections)} objects on synthetic frame.")
        except Exception as e:
            print(f"  Real YOLO initialization failed: {e}")

        if not detections:
            print("  [FAKE YOLO] Using synthetic fallback detections (simulated person & car):")
            detections = [
                Detection(label="person", direction="FRONT", confidence=0.88, box=(300, 200, 500, 600)),
                Detection(label="car", direction="FRONT", confidence=0.92, box=(700, 350, 1100, 650)),
            ]
            for d in detections:
                print(f"    - FAKE Detection: {d.label} at box={d.box}, conf={d.confidence}")

        # 2. Depth Stage
        try:
            from perception.mono_depth import MonoDepth
            md = MonoDepth()
            # Do not download 100MB HF weights in headless test: use heuristic fallback
            boxes = [d.box for d in detections]
            labels = [d.label for d in detections]
            print("  Running MonoDepth heuristic fallback...")
            depths = md._heuristic(boxes, labels)
            for d, dm in zip(detections, depths):
                d.distance_m = dm
                d.depth_mode = "heuristic"
                print(f"  Estimated depth for {d.label}: {d.distance_m}m ({d.depth_mode})")
        except Exception as e:
            print(f"  Depth estimation stage failed: {e}")
            if first_failure is None:
                first_failure = f"Stage B: Depth estimation error ({e})"

        # 3. Tracking Stage
        try:
            from perception.tracker import ObjectTracker
            tracker = ObjectTracker()
            tracked_dets = tracker.update(detections, img)
            print(f"  ObjectTracker update passed. Tracks count: {len(tracker.active_tracks)}")
            for d in tracked_dets:
                print(f"    - Tracked: {d.label} track_id={d.track_id}")
        except Exception as e:
            print(f"  Tracking stage failed: {e}")
            if first_failure is None:
                first_failure = f"Stage B: Tracking error ({e})"

        # 4. Egomotion Stage
        try:
            from perception.egomotion import OpticalFlowEgomotion
            import cv2
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ego = OpticalFlowEgomotion()
            speed, state = ego.update(gray, [d.box for d in detections], [d.label for d in detections])
            print(f"  OpticalFlowEgomotion passed: user_speed={speed} m/s, state='{state}'")
        except Exception as e:
            print(f"  Egomotion stage failed: {e}")
            if first_failure is None:
                first_failure = f"Stage B: Egomotion error ({e})"

        # 5. Kinematics (Speed & TTC) Stage
        try:
            from kinematics.speed import SpeedEstimator
            from kinematics.ttc import TTCCalculator
            se = SpeedEstimator()
            ttc_calc = TTCCalculator()
            for d in tracked_dets:
                trk = tracker.active_tracks.get(d.track_id)
                if trk:
                    # Provide second history entry to simulate motion
                    trk.history.append((d.distance_m - 0.5, (400, 400), time.time() + 0.1))
                    spd, mot = se.update(trk, 0.0, "Stationary")
                    d.speed_mps = spd
                    d.motion = mot
                    d.ttc_sec = ttc_calc.compute(d.distance_m, spd, mot, None)
                    print(f"    - Kinematics: {d.label} speed={spd}m/s, motion={mot}, ttc={d.ttc_sec}s, threat_score={d.threat_score:.1f}")
        except Exception as e:
            print(f"  Kinematics stage failed: {e}")
            if first_failure is None:
                first_failure = f"Stage B: Kinematics error ({e})"

        # 6. Priority Queue Stage
        try:
            from core.priority_queue import DetectionPriorityQueue
            pq = DetectionPriorityQueue()
            pq.push_all(tracked_dets)
            sorted_dets = pq.drain()
            print(f"  PriorityQueue passed. Drained order: {[d.label for d in sorted_dets]}")
        except Exception as e:
            print(f"  Priority Queue stage failed: {e}")
            if first_failure is None:
                first_failure = f"Stage B: Priority Queue error ({e})"

        # 7. Grouping & Speech Messages Stage
        try:
            from kinematics.heatmap import group_detections, build_speech_messages
            grouped = group_detections(sorted_dets)
            hp = [d for d in sorted_dets if d.is_high_priority]
            msgs = build_speech_messages(grouped, hp)
            print(f"  Heatmap message builder passed. Spoken messages: {msgs}")
        except Exception as e:
            print(f"  Speech message building stage failed: {e}")
            if first_failure is None:
                first_failure = f"Stage B: Heatmap / speech message error ({e})"

    except Exception as e:
        print(f"  Synthetic pipeline crashed: {e}")
        traceback.print_exc()
        if first_failure is None:
            first_failure = f"Stage B: Pipeline crash ({e})"

    # -------------------------------------------------------------------------
    # STAGE C: Agent Tools with Dummy Inputs and Mock LLM
    # -------------------------------------------------------------------------
    print("\n[STAGE C] AGENT TOOLS EXECUTION TEST (WITH MOCK LLM)")
    print("-" * 50)
    
    # We test each tool individually
    tools_tested = [
        ("set_persistent_goal", lambda t: t.invoke({"user_need": "water"})),
        ("mark_goal_completed", lambda t: t.invoke({"goal_or_object": "water"})),
        ("lower_goal_candidate_priority", lambda t: t.invoke({"candidate": "bottle"})),
        ("query_past_detections", lambda t: t.invoke({"question": "where is my water bottle?"})),
        ("summarize_scene", lambda t: t.invoke({"scene_description": "person at 2m in front"})),
        ("set_search_intent", lambda t: t.invoke({"object_types": "chair"})),
        ("save_object_signature", lambda t: t.invoke({"label": "bottle", "alias": "my water", "current_spatial_context": "at 1m in front"})),
        ("get_objects_near", lambda t: t.invoke({})),
    ]

    # Try importing tools directly
    vt_module = None
    try:
        import agents.tools.vision_tools as vt
        vt_module = vt
        print("  Imported agents.tools.vision_tools directly without error.")
    except Exception as e:
        print(f"  Failed to import agents.tools.vision_tools directly: {type(e).__name__}: {e}")
        if first_failure is None:
            first_failure = f"Stage C: Failed to import vision_tools ({e})"

    if vt_module:
        for tool_name, caller in tools_tested:
            if hasattr(vt_module, tool_name):
                tool_obj = getattr(vt_module, tool_name)
                try:
                    res = caller(tool_obj)
                    print(f"  [PASS] Tool '{tool_name}': {res}")
                except Exception as e:
                    print(f"  [FAIL] Tool '{tool_name}': {type(e).__name__}: {e}")
                    if first_failure is None:
                        first_failure = f"Stage C: Tool execution error in {tool_name} ({e})"
            else:
                print(f"  [MISSING] Tool '{tool_name}' not defined in vision_tools.py")
                if first_failure is None:
                    first_failure = f"Stage C: Missing tool {tool_name}"

    # -------------------------------------------------------------------------
    # STAGE D: Summary of First Failure & End-to-End Status
    # -------------------------------------------------------------------------
    print("\n[STAGE D] HEADLESS SMOKE TEST SUMMARY")
    print("-" * 50)
    print(f"First failure point identified: {first_failure if first_failure else 'NONE (All stages passed)'}")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
