"""
Unit tests for the pure math helper functions used across the pipeline:
compute_iou (tracker.py), euclidean_distance/get_center (motion.py),
and box_distance (interaction.py).

These are pure functions (no side effects, no I/O), so they're tested
with hand-picked inputs where the correct answer can be verified by hand.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.tracker import compute_iou
from pipeline.motion import get_center, euclidean_distance
from pipeline.interaction import box_distance


class TestComputeIoU:
    def test_identical_boxes_have_iou_of_1(self):
        box = [10, 10, 50, 50]
        assert compute_iou(box, box) == 1.0

    def test_non_overlapping_boxes_have_iou_of_0(self):
        box_a = [0, 0, 10, 10]
        box_b = [100, 100, 110, 110]
        assert compute_iou(box_a, box_b) == 0.0

    def test_partial_overlap_computed_correctly(self):
        # box_a: 0-20 x 0-20 (area 400)
        # box_b: 10-30 x 0-20 (area 400)
        # intersection: 10-20 x 0-20 = 10*20 = 200
        # union: 400 + 400 - 200 = 600
        # IoU = 200/600 = 0.333...
        box_a = [0, 0, 20, 20]
        box_b = [10, 0, 30, 20]
        result = compute_iou(box_a, box_b)
        assert math.isclose(result, 1 / 3, rel_tol=1e-6)

    def test_touching_edges_has_iou_of_0(self):
        # Boxes that share an edge but don't overlap in area
        box_a = [0, 0, 10, 10]
        box_b = [10, 0, 20, 10]
        assert compute_iou(box_a, box_b) == 0.0


class TestEuclideanDistance:
    def test_same_point_has_distance_0(self):
        assert euclidean_distance((5, 5), (5, 5)) == 0.0

    def test_known_3_4_5_triangle(self):
        # Classic 3-4-5 right triangle -> distance should be exactly 5
        assert euclidean_distance((0, 0), (3, 4)) == 5.0

    def test_get_center_of_box(self):
        # Box from (0,0) to (10,20) -> center should be (5, 10)
        assert get_center([0, 0, 10, 20]) == (5.0, 10.0)


class TestBoxDistance:
    def test_overlapping_boxes_have_distance_0(self):
        box_a = [0, 0, 20, 20]
        box_b = [10, 10, 30, 30]
        assert box_distance(box_a, box_b) == 0.0

    def test_touching_boxes_have_distance_0(self):
        box_a = [0, 0, 10, 10]
        box_b = [10, 0, 20, 10]
        assert box_distance(box_a, box_b) == 0.0

    def test_horizontally_separated_boxes(self):
        # box_a right edge at x=10, box_b left edge at x=15 -> gap of 5
        box_a = [0, 0, 10, 10]
        box_b = [15, 0, 25, 10]
        assert box_distance(box_a, box_b) == 5.0

    def test_diagonally_separated_boxes(self):
        # box_a: (0,0)-(10,10). box_b: (13,14)-(20,20).
        # horizontal gap = 13-10 = 3, vertical gap = 14-10 = 4
        # distance = sqrt(3^2 + 4^2) = 5 (3-4-5 triangle again)
        box_a = [0, 0, 10, 10]
        box_b = [13, 14, 20, 20]
        assert box_distance(box_a, box_b) == 5.0