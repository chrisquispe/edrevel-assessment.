"""
detector.py

Wraps YOLO (via the ultralytics library) to detect objects in a single
frame. This module's only responsibility is: "given one image, what
objects are in it, where, and how confident are we?" It doesn't know
about tracking across frames, motion, or interactions -- those are
separate modules, which is what "modular pipeline" means.
"""

from ultralytics import YOLO

# Load the pretrained YOLO model once, at import time -- not inside the
# detect function. Loading the model is slow (it reads weights from
# disk); we want to pay that cost ONCE per app run, not once per frame.
#
# "yolov8n.pt" is the "nano" variant -- the smallest, fastest YOLOv8
# model. It trades some accuracy for speed, which is a reasonable choice
# for an assessment that needs to process video quickly. This is
# documented as a tradeoff in the README.
_model = YOLO("yolov8n.pt")


def detect_objects(frame, confidence_threshold: float = 0.35):
    """
    Runs YOLO on a single frame (a numpy array, RGB, as produced by
    frame_extractor.extract_frames).

    Returns a list of detections, each a dict:
        {
            "class_name": str,       e.g. "person", "laptop"
            "confidence": float,     0.0 to 1.0
            "bbox": [x1, y1, x2, y2] # bounding box corners, in pixels
        }

    confidence_threshold: detections below this score are discarded.
    YOLO will happily report very low-confidence guesses; filtering
    keeps only detections we can reasonably trust.
    """
    results = _model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        confidence = float(box.conf[0])
        if confidence < confidence_threshold:
            continue

        class_id = int(box.cls[0])
        class_name = _model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    return detections