# main.py
import os
import cv2
from collections import defaultdict

from ultralytics import YOLO

from config import *
from tracker import new_track
from utils import *
from fall_analyzer import analyze_fall_indicators


# ==========================================================
# SETUP
# ==========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

det_model = YOLO(MODEL_PATH_DETECTION)
pose_model = YOLO(MODEL_PATH_POSE)


# ==========================================================
# VIDEO PROCESSING
# ==========================================================

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    track_memory = defaultdict(new_track)
    prev_boxes = {}
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # ---------------- DETECTION ----------------
        det = det_model(frame, conf=CONF_THRESH, classes=[0], verbose=False)[0]
        pose = pose_model(frame, conf=POSE_CONF_THRESH, verbose=False)[0]

        curr_boxes = {}
        curr_poses = {}

        if det.boxes is not None:
            for i, box in enumerate(det.boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                if (x2 - x1) * (y2 - y1) / (W * H) > MIN_AREA_RATIO:
                    curr_boxes[i] = (x1, y1, x2, y2)

        if pose.keypoints is not None:
            for i, kp in enumerate(pose.keypoints):
                if i not in curr_boxes:
                    continue
                pts = kp.xy.cpu().numpy()[0]
                conf = kp.conf.cpu().numpy()[0]
                curr_poses[i] = {
                    j: (pts[j][0], pts[j][1])
                    for j in range(len(pts)) if conf[j] > 0.5
                }

        # ---------------- TRACK MATCH ----------------
        matches = match_tracks(prev_boxes, curr_boxes)
        assignments = {}
        next_id = max(track_memory.keys()) + 1 if track_memory else 0

        for cid in curr_boxes:
            assigned = False
            for tid, mid in matches.items():
                if mid == cid:
                    assignments[tid] = cid
                    assigned = True
                    break
            if not assigned:
                assignments[next_id] = cid
                next_id += 1

        updated_boxes = {}

        # ---------------- TRACK UPDATE ----------------
        for tid, cid in assignments.items():
            x1, y1, x2, y2 = curr_boxes[cid]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            mem = track_memory[tid]
            keypoints = curr_poses.get(cid, {})

            mem["centers"].append((cx, cy))
            mem["aspect_ratios"].append(calculate_aspect_ratio((x1, y1, x2, y2)))
            mem["keypoints"].append(keypoints)
            mem["last_seen_y"] = cy

            angle = calculate_body_angle(keypoints)
            if angle is not None:
                mem["body_angles"].append(angle)

            # velocity / acceleration
            if len(mem["centers"]) >= 2:
                v = compute_vertical_velocity(
                    mem["centers"][-2][1], mem["centers"][-1][1]
                )
                mem["vel_y"].append(v)

                if len(mem["vel_y"]) >= 2:
                    a = compute_vertical_acceleration(
                        mem["vel_y"][-2], mem["vel_y"][-1]
                    )
                    mem["acc_y"].append(a)

            # stability
            if len(mem["centers"]) >= 2:
                mem["center_var"].append(abs(mem["centers"][-1][1] - mem["centers"][-2][1]))

            if len(mem["body_angles"]) >= 2:
                mem["angle_var"].append(abs(mem["body_angles"][-1] - mem["body_angles"][-2]))

            if keypoints:
                mem["is_human"] = is_valid_human_pose(keypoints)

            mem["frames_seen"] += 1
            mem["missing"] = 0

            updated_boxes[tid] = (x1, y1, x2, y2)

        # ---------------- MISSING TRACKS ----------------
        for tid in list(track_memory.keys()):
            if tid not in updated_boxes:
                mem = track_memory[tid]
                mem["missing"] += 1
                mem["occlusion_frames"] += 1
                if mem["missing"] > MAX_MISSING:
                    del track_memory[tid]

        # ---------------- FSM ----------------
        for tid, mem in track_memory.items():
            if not mem["is_human"] or mem["locked"]:
                continue

            signals = analyze_fall_indicators(mem, H)

            if mem["state"] == "standing":
                if signals["fall_evidence"]:
                    mem["state"] = "falling"
                    mem["fall_votes"] = 1

            elif mem["state"] == "falling":
                if signals["fall_evidence"]:
                    mem["fall_votes"] += 1
                else:
                    mem["fall_votes"] = max(0, mem["fall_votes"] - 1)

                if mem["fall_votes"] >= FALL_CONFIRM_FRAMES and (
                    signals["near_ground"] or signals["hidden"]
                ):
                    mem["state"] = "fallen"
                    mem["fall_detected"] = True

            elif mem["state"] == "fallen":
                mem["ground_time"] += 1
                if signals.get("inactive", False):
                    mem["locked"] = True

        # ---------------- DRAW ----------------
        for tid, box in updated_boxes.items():
            mem = track_memory[tid]
            if not mem["is_human"]:
                continue

            x1, y1, x2, y2 = box
            color = COLOR_STANDING
            label = "STANDING"

            if mem["state"] == "falling":
                color = COLOR_FALLING
                label = "FALLING"
            elif mem["state"] == "fallen":
                color = COLOR_FALLEN
                label = "FALLEN"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"ID:{tid} {label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        writer.write(frame)
        prev_boxes = updated_boxes.copy()

        if frame_id % 30 == 0:
            print(f"Processed {frame_id} frames")

    cap.release()
    writer.release()
    print("Finished:", output_path)


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    for v in os.listdir(INPUT_DIR):
        if v.lower().endswith((".mp4", ".avi", ".mov")):
            process_video(
                os.path.join(INPUT_DIR, v),
                os.path.join(OUTPUT_DIR, v)
            )
