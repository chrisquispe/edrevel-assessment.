"""
interaction.py

Determines which frames a person is "interacting" with an object, based
on proximity between the person's bounding box and the object's
bounding box.

Approach: rather than requiring literal pixel overlap (too strict --
a person's whole-body box might not perfectly overlap a small object
they're reaching for), we measure the GAP between the two boxes. If
the gap is small enough (or they overlap), we count it as interaction.
This is the same tradeoff noted in motion.py: we're using the
whole-body box as a stand-in for hand position, since standard YOLO
doesn't detect hands directly.
"""


def box_distance(box_a, box_b):
    """
    Computes the shortest distance between two bounding boxes' edges.

    If the boxes overlap at all, returns 0.0 (they're touching/inside
    each other). Otherwise, returns the pixel gap between their nearest
    edges.

    Each box is [x1, y1, x2, y2].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Horizontal gap: how far apart are the boxes left-to-right?
    # If they overlap horizontally, this is 0.
    if ax2 < bx1:
        dx = bx1 - ax2       # box_a is entirely to the left of box_b
    elif bx2 < ax1:
        dx = ax1 - bx2       # box_a is entirely to the right of box_b
    else:
        dx = 0.0             # they overlap horizontally

    # Vertical gap: same idea, top-to-bottom.
    if ay2 < by1:
        dy = by1 - ay2       # box_a is entirely above box_b
    elif by2 < ay1:
        dy = ay1 - by2       # box_a is entirely below box_b
    else:
        dy = 0.0             # they overlap vertically

    # If both gaps are 0, the boxes overlap -> distance is 0.
    # Otherwise, combine the two gaps into a single distance
    # (Pythagorean theorem, same idea as euclidean_distance elsewhere).
    return (dx ** 2 + dy ** 2) ** 0.5


def detect_interactions(person_frames, object_frames, proximity_threshold: float = 40.0, gap_tolerance: int = 5):
    """
    gap_tolerance: bridges over brief stretches of missing detections
    (e.g. the object detector missing a frame or two) so one continuous
    real-world interaction doesn't get artificially split into several
    tiny intervals. Frames separated by `gap_tolerance` frames or fewer
    are treated as part of the same interaction.
    """
    interacting_frames = []
    for frame_index, obj_bbox in sorted(object_frames.items()):
        if frame_index not in person_frames:
            continue
        person_bbox = person_frames[frame_index]
        distance = box_distance(person_bbox, obj_bbox)
        if distance <= proximity_threshold:
            interacting_frames.append(frame_index)

    if not interacting_frames:
        return []

    intervals = []
    range_start = interacting_frames[0]
    prev_frame = range_start
    for frame_index in interacting_frames[1:]:
        if frame_index - prev_frame > gap_tolerance:
            intervals.append({"interacted_by_person": 0, "frame_start": range_start, "frame_end": prev_frame})
            range_start = frame_index
        prev_frame = frame_index
    intervals.append({"interacted_by_person": 0, "frame_start": range_start, "frame_end": prev_frame})
    return intervals