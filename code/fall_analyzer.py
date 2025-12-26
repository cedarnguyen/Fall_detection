"""
fall_analyzer.py
Evidence-based fall analysis (FINAL)
"""

import numpy as np


def posture_collapse(mem, angle_thresh=45, ratio_thresh=1.0):
    if not mem["body_angles"] or not mem["aspect_ratios"]:
        return False
    return (
        mem["body_angles"][-1] > angle_thresh or
        mem["aspect_ratios"][-1] < ratio_thresh
    )


def motion_evidence(mem, vel_thresh=25, acc_thresh=20):
    if mem["vel_y"] and max(mem["vel_y"]) > vel_thresh:
        return True
    if mem["acc_y"] and max(mem["acc_y"]) > acc_thresh:
        return True
    return False


def stability_lost(mem, var_thresh=5):
    if mem["center_var"] and max(mem["center_var"]) > var_thresh:
        return True
    if mem["angle_var"] and max(mem["angle_var"]) > var_thresh:
        return True
    return False


def near_ground(mem, frame_h, ratio=0.75):
    return mem["last_seen_y"] is not None and mem["last_seen_y"] > frame_h * ratio


def hidden_fall(mem, frame_h):
    return (
        mem["state"] == "falling" and
        mem["occlusion_frames"] >= 5 and
        near_ground(mem, frame_h, 0.65)
    )


def analyze_fall_indicators(mem, frame_h):
    # Independent evidence groups
    posture = posture_collapse(mem)          # body form
    motion = motion_evidence(mem)             # velocity / acceleration
    unstable = stability_lost(mem)            # variance / loss of control

    near = near_ground(mem, frame_h)
    hidden = hidden_fall(mem, frame_h)

    # --------------------------------------------------
    # CONSENSUS RULE (CRITICAL)
    # --------------------------------------------------
    # At least TWO evidences must be true
    # AND posture or instability must be one of them
    evidence_count = sum([posture, motion, unstable])

    fall_evidence = (
        evidence_count >= 2
        and (posture or unstable)
    )

    return {
        "posture": posture,
        "motion": motion,
        "unstable": unstable,
        "near_ground": near,
        "hidden": hidden,
        "fall_evidence": fall_evidence
    }

