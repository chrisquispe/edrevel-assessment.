"""
motion.py

Classifies each tracked object as "moving" or "stationary" over time,
based on how much its position changes.

Approach: for each frame, compare the object's current center point to
its center point WINDOW frames earlier (not just the immediately
previous frame). Comparing over a wider time window catches gradual,
sustained movement (e.g. someone slowly leaning over a bench) that
frame-to-frame comparison misses, since tiny per-frame shifts can
individually fall under the noise threshold while adding up to real
movement over time. Consecutive same-state frames are merged into
ranges, matching the required motion_history schema.
"""

import math


def get_center(bbox):
    """Given [x1, y1, x2, y2], returns the box's center point (x, y)."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def euclidean_distance(point_a, point_b):
    """Straight-line distance between two (x, y) points."""
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def classify_motion_states(frame_bbox_pairs: list, movement_threshold: float = 20.0, window: int = 12):
    """
    frame_bbox_pairs: list of (frame_index, bbox) tuples for ONE tracked
    object, in increasing frame order.

    movement_threshold: minimum pixel distance (measured across `window`
    frames) to count as "moving". Tuned empirically against the sample
    video -- large enough to ignore per-frame detection jitter, small
    enough to catch real repositioning.

    window: how many frames back to compare against. E.g. window=8 at
    24fps compares each frame to roughly 1/3 second earlier, so slow,
    gradual movement (that a 1-frame comparison would miss) still gets
    detected once it accumulates across the window.

    Returns merged motion_history segments:
        [{"frame_range": [0, 86], "state": "stationary"}, ...]
    """
    if len(frame_bbox_pairs) < 2:
        frame_index = frame_bbox_pairs[0][0] if frame_bbox_pairs else 0
        return [{"frame_range": [frame_index, frame_index], "state": "stationary"}]

    per_frame_states = []

    for i in range(len(frame_bbox_pairs)):
        curr_frame_index, curr_bbox = frame_bbox_pairs[i]

        # Compare against `window` observations back, not just the
        # previous one. Clamp to index 0 for the first few frames where
        # a full window isn't available yet.
        compare_index = max(0, i - window)
        _, compare_bbox = frame_bbox_pairs[compare_index]

        distance = euclidean_distance(get_center(compare_bbox), get_center(curr_bbox))
        state = "moving" if distance > movement_threshold else "stationary"
        per_frame_states.append((curr_frame_index, state))

    # Merge consecutive same-state frames into ranges.
    merged = []
    range_start_frame, current_state = per_frame_states[0]
    prev_frame_index = range_start_frame

    for frame_index, state in per_frame_states[1:]:
        if state != current_state:
            merged.append({"frame_range": [range_start_frame, prev_frame_index], "state": current_state})
            range_start_frame = frame_index
            current_state = state
        prev_frame_index = frame_index

    merged.append({"frame_range": [range_start_frame, prev_frame_index], "state": current_state})
    return merged