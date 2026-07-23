"""
frame_extractor.py

Responsible for one job: opening a video file and yielding its frames
one at a time, along with basic metadata (fps, resolution, total frame
count). Every other pipeline module (detector, tracker, motion, etc.)
consumes frames produced here -- this keeps "reading video" separate
from "understanding what's in the video," which is what the assessment
means by a "modular pipeline."
"""

import cv2


def get_video_metadata(video_path: str) -> dict:
    """
    Opens the video just long enough to read its metadata, then closes it.
    Returns duration, frame count, resolution, and fps -- this maps
    directly to the `videoMetadata` field in the required JSON schema.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "duration_seconds": round(duration_seconds, 2),
        "frame_count": frame_count,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 2),
    }


def extract_frames(video_path: str, frame_skip: int = 1):
    """
    A generator that yields (frame_index, frame_image) tuples for the
    given video.

    `frame_skip`: process every Nth frame instead of every single one.
    Why this matters: at 24fps, an 8-second video has ~192 frames. Running
    a deep learning model on every single frame is the slowest part of
    this whole pipeline. Skipping frames (e.g. frame_skip=2 processes
    every other frame) trades a little temporal precision for a lot of
    speed -- a reasonable, explainable tradeoff worth mentioning in the
    README. Default of 1 processes every frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break  # end of video

        if frame_index % frame_skip == 0:
            # OpenCV reads frames in BGR color order (blue-green-red)
            # instead of the more common RGB. YOLO expects RGB, so we
            # convert here, once, in the one place frames enter the
            # pipeline -- rather than every downstream module having to
            # remember to do it.
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame_index, frame_rgb

        frame_index += 1

    cap.release()