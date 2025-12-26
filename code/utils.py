# utils.py
import math
import numpy as np


# ==========================================================
# GEOMETRY
# ==========================================================

def calculate_aspect_ratio(bbox):
    """
    bbox = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = bbox
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return h / w


# ==========================================================
# POSE / BODY ANGLE
# ==========================================================

def calculate_body_angle(keypoints):
    """
    Estimate torso tilt angle (degrees) using shoulders & hips.
    Returns None if insufficient keypoints.
    """
    # COCO indices
    L_SHOULDER, R_SHOULDER = 5, 6
    L_HIP, R_HIP = 11, 12

    try:
        sx = (keypoints[L_SHOULDER][0] + keypoints[R_SHOULDER][0]) / 2
        sy = (keypoints[L_SHOULDER][1] + keypoints[R_SHOULDER][1]) / 2
        hx = (keypoints[L_HIP][0] + keypoints[R_HIP][0]) / 2
        hy = (keypoints[L_HIP][1] + keypoints[R_HIP][1]) / 2
    except Exception:
        return None

    dx = hx - sx
    dy = hy - sy

    if dy == 0:
        return 90.0

    angle = abs(math.degrees(math.atan(dx / dy)))
    return angle


# ==========================================================
# MOTION
# ==========================================================

def compute_vertical_velocity(prev_y, curr_y):
    return curr_y - prev_y


def compute_vertical_acceleration(prev_v, curr_v):
    return curr_v - prev_v


# ==========================================================
# TRACK MATCHING (IoU)
# ==========================================================

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def match_tracks(prev_boxes, curr_boxes, iou_thresh=0.3):
    """
    Simple greedy IoU matching
    """
    matches = {}
    used = set()

    for pid, pbox in prev_boxes.items():
        best_iou, best_cid = 0, None
        for cid, cbox in curr_boxes.items():
            if cid in used:
                continue
            v = iou(pbox, cbox)
            if v > best_iou:
                best_iou, best_cid = v, cid

        if best_iou > iou_thresh:
            matches[pid] = best_cid
            used.add(best_cid)

    return matches


# ==========================================================
# HUMAN VALIDATION
# ==========================================================

def is_valid_human_pose(keypoints, min_points=6):
    """
    Reject mannequins / false detections
    """
    return len(keypoints) >= min_points
