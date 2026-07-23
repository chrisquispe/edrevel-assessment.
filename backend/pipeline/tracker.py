"""
tracker.py

Assigns a consistent numeric object_id to detections across frames, so
the same physical cable/person/device keeps the same ID from frame 1
to frame 192, instead of being treated as a brand-new object every
single frame.

Approach: IoU (Intersection over Union) matching. Between consecutive
frames, objects move only slightly (video runs at 24 frames/second), so
a detection's bounding box in frame N will overlap heavily with the
same object's box in frame N-1. We match new detections to existing
tracked objects by finding the highest-overlap pair.

This is a simplified version of the standard "SORT" tracking algorithm
-- intentionally kept minimal and readable rather than pulling in an
external tracking library, per the assessment's ask for "easily
understandable mathematical helper functions."
"""


def compute_iou(box_a, box_b):
    """
    Computes Intersection over Union between two bounding boxes.

    Each box is [x1, y1, x2, y2] -- top-left and bottom-right corners,
    in pixels.

    Returns a float between 0.0 (no overlap) and 1.0 (identical boxes).
    """
    # Find the coordinates of the overlapping rectangle (if any).
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[2] if False else box_b[3])

    # If the boxes don't overlap at all, width or height would be
    # negative -- clamp to 0 in that case.
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection_area = inter_width * inter_height

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


class ObjectTracker:
    """
    Maintains state across frames. Call `update()` once per frame with
    that frame's detections; it returns those same detections annotated
    with a consistent `object_id`.
    """

    def __init__(self, iou_threshold: float = 0.25, max_frames_missing: int = 35):
        # object." Lower = more lenient matching, higher = stricter.
        self.iou_threshold = iou_threshold

        # If a tracked object isn't matched for this many consecutive
        # frames, we consider it gone (e.g. it left the frame) and stop
        # trying to match new detections to it.
        self.max_frames_missing = max_frames_missing

        self._next_id = 1
        # Active tracks: object_id -> {"bbox": [...], "class_name": str, "frames_missing": int}
        self._tracks = {}

    def update(self, detections: list, frame_index: int):
        """
        detections: list of {"class_name", "confidence", "bbox"} from
        detector.detect_objects() for the CURRENT frame.

        Returns the same list, with an added "object_id" key on each.
        """
        matched_track_ids = set()
        annotated_detections = []

        for detection in detections:
            best_match_id = None
            best_iou = 0.0

            # Compare this detection against every currently active
            # track, keep the best-overlapping one.
            for track_id, track in self._tracks.items():
                if track_id in matched_track_ids:
                    continue  # each track can only match one detection per frame
                if track["class_name"] != detection["class_name"]:
                    continue  # a "person" should never match a "bottle"

                iou = compute_iou(track["bbox"], detection["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id

            if best_match_id is not None and best_iou >= self.iou_threshold:
                # Same object as an existing track -- reuse its ID.
                object_id = best_match_id
            else:
                # No good match -- this is a new object.
                object_id = self._next_id
                self._next_id += 1

            # Update (or create) the track record.
            self._tracks[object_id] = {
                "bbox": detection["bbox"],
                "class_name": detection["class_name"],
                "frames_missing": 0,
            }
            matched_track_ids.add(object_id)

            annotated_detections.append({**detection, "object_id": object_id})

        # Any track NOT matched this frame gets older. Drop it entirely
        # if it's been missing too long (object likely left the frame).
        stale_ids = []
        for track_id, track in self._tracks.items():
            if track_id not in matched_track_ids:
                track["frames_missing"] += 1
                if track["frames_missing"] > self.max_frames_missing:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            del self._tracks[track_id]

        return annotated_detections