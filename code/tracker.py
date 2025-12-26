"""
tracker.py - Track memory and state management
"""
from collections import deque, defaultdict

def new_track():
    return {
        # Geometry
        "centers": deque(maxlen=30),
        "heights": deque(maxlen=15),
        "aspect_ratios": deque(maxlen=15),

        # Pose
        "keypoints": deque(maxlen=30),
        "body_angles": deque(maxlen=10),

        # Motion
        "vel_y": deque(maxlen=10),
        "acc_y": deque(maxlen=10),
        "motion_energy": deque(maxlen=20),

        # Body parts
        "part_centers": deque(maxlen=30),
        "part_velocities": defaultdict(lambda: deque(maxlen=10)),

        # State
        "frames_seen": 0,
        "missing": 0,
        "valid": False,
        "is_human": False,
        "is_mannequin": False,

        # Occlusion
        "occluded": False,
        "occlusion_frames": 0,
        "last_seen_y": None,

        # Fall FSM
        "state": "standing",          # standing | falling | occluded | fallen
        "fall_votes": 0,
        "impulse_detected": False,
        "ground_time": 0,
        "fall_detected": False,
        "locked": False,
        "angle_var": deque(maxlen=15),
        "center_var": deque(maxlen=15),

    }


def new_body_part_track(parent_id, part_name):
    """Create a sub-track for a body part"""
    return {
        "parent_id": parent_id,
        "part_name": part_name,
        "centers": deque(maxlen=30),
        "velocities": deque(maxlen=10),
        "frames_seen": 0,
        "missing": 0,
        "last_seen_y": None,
        "fall_detected": False,
    }