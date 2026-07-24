"""
Tests for the higher-level logic functions: classify_motion_states
(motion.py) and detect_interactions (interaction.py). These use small,
hand-crafted synthetic scenarios with a known correct answer, rather
than the real video, so they run instantly and don't depend on YOLO.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.motion import classify_motion_states
from pipeline.interaction import detect_interactions


class TestClassifyMotionStates:
    def test_stationary_object_stays_in_one_place(self):
        # Same bbox across every frame -> should be one long "stationary" segment
        frames = [(i, [100, 100, 150, 150]) for i in range(10)]
        result = classify_motion_states(frames, movement_threshold=20.0, window=1)
        assert len(result) == 1
        assert result[0]["state"] == "stationary"
        assert result[0]["frame_range"] == [0, 9]

    def test_object_that_moves_far_is_classified_moving(self):
        # Object jumps 500px between frame 0 and frame 1 -- clearly moving
        frames = [
            (0, [0, 0, 50, 50]),
            (1, [500, 500, 550, 550]),
        ]
        result = classify_motion_states(frames, movement_threshold=20.0, window=1)
        # Frame 0 has no prior frame to compare to -> defaults stationary.
        # Frame 1 moved far from frame 0 -> moving.
        states = [seg["state"] for seg in result]
        assert "moving" in states

    def test_single_frame_defaults_to_stationary(self):
        frames = [(0, [10, 10, 20, 20])]
        result = classify_motion_states(frames)
        assert len(result) == 1
        assert result[0]["state"] == "stationary"

    def test_small_jitter_under_threshold_stays_stationary(self):
        # Tiny 2px shifts between frames -- should NOT count as moving
        # when threshold is 20px.
        frames = [
            (0, [100, 100, 150, 150]),
            (1, [101, 101, 151, 151]),
            (2, [102, 100, 152, 150]),
            (3, [100, 102, 150, 152]),
        ]
        result = classify_motion_states(frames, movement_threshold=20.0, window=1)
        assert all(seg["state"] == "stationary" for seg in result)


class TestDetectInteractions:
    def test_no_overlap_produces_no_interactions(self):
        person_frames = {0: [0, 0, 50, 50], 1: [0, 0, 50, 50]}
        object_frames = {0: [500, 500, 550, 550], 1: [500, 500, 550, 550]}
        result = detect_interactions(person_frames, object_frames, proximity_threshold=40.0)
        assert result == []

    def test_overlapping_boxes_produce_one_interaction(self):
        person_frames = {0: [0, 0, 50, 50], 1: [0, 0, 50, 50], 2: [0, 0, 50, 50]}
        object_frames = {0: [10, 10, 30, 30], 1: [10, 10, 30, 30], 2: [10, 10, 30, 30]}
        result = detect_interactions(person_frames, object_frames, proximity_threshold=40.0)
        assert len(result) == 1
        assert result[0]["frame_start"] == 0
        assert result[0]["frame_end"] == 2
        assert result[0]["interacted_by_person"] == 0

    def test_gap_tolerance_bridges_brief_missing_detections(self):
        # Object detected at frames 0-2 and 5-7 (gap at 3,4), all
        # touching the person. gap_tolerance=5 should merge these into
        # ONE interaction interval instead of two.
        person_frames = {i: [0, 0, 50, 50] for i in range(10)}
        object_frames = {i: [10, 10, 30, 30] for i in [0, 1, 2, 5, 6, 7]}
        result = detect_interactions(person_frames, object_frames, proximity_threshold=40.0, gap_tolerance=5)
        assert len(result) == 1
        assert result[0]["frame_start"] == 0
        assert result[0]["frame_end"] == 7

    def test_large_gap_produces_separate_intervals(self):
        # Same as above, but gap is now 20 frames -- exceeds
        # gap_tolerance=5, so should produce TWO separate intervals.
        person_frames = {i: [0, 0, 50, 50] for i in range(30)}
        object_frames = {i: [10, 10, 30, 30] for i in [0, 1, 2, 25, 26, 27]}
        result = detect_interactions(person_frames, object_frames, proximity_threshold=40.0, gap_tolerance=5)
        assert len(result) == 2

    def test_person_not_visible_in_frame_is_skipped(self):
        # Object appears in frame 5, but the person has no data for
        # frame 5 -- should not crash, and should not count that frame.
        person_frames = {0: [0, 0, 50, 50]}
        object_frames = {0: [10, 10, 30, 30], 5: [10, 10, 30, 30]}
        result = detect_interactions(person_frames, object_frames, proximity_threshold=40.0)
        assert len(result) == 1
        assert result[0]["frame_end"] == 0