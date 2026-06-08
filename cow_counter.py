"""Cow detection helper backed by a pretrained YOLOv4-tiny (COCO) model."""
import os

import cv2
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.weights")
CONFIG_PATH = os.path.join(MODELS_DIR, "yolov4-tiny.cfg")
NAMES_PATH = os.path.join(MODELS_DIR, "coco.names")

CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.4
INPUT_SIZE = 416

with open(NAMES_PATH, encoding="utf-8") as f:
    _CLASS_NAMES = [line.strip() for line in f if line.strip()]
COW_CLASS_ID = _CLASS_NAMES.index("cow")

_net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
_OUTPUT_LAYERS = _net.getUnconnectedOutLayersNames()


def detect_cows(image_bgr):
    """Run cow detection on a BGR image (numpy array).

    Returns a tuple of (count, boxes) where boxes is a list of
    (x, y, w, h, confidence) for each detected cow, after NMS.
    """
    height, width = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image_bgr, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    _net.setInput(blob)
    layer_outputs = _net.forward(_OUTPUT_LAYERS)

    boxes = []
    confidences = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if class_id != COW_CLASS_ID or confidence < CONFIDENCE_THRESHOLD:
                continue

            center_x, center_y, box_w, box_h = (
                detection[0:4] * np.array([width, height, width, height])
            )
            x = int(center_x - box_w / 2)
            y = int(center_y - box_h / 2)
            boxes.append([x, y, int(box_w), int(box_h)])
            confidences.append(confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    indices = np.array(indices).flatten() if len(indices) else []

    final_boxes = [
        (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], confidences[i])
        for i in indices
    ]
    return len(final_boxes), final_boxes


def draw_detections(image_bgr, boxes):
    """Return a copy of the image with bounding boxes drawn around detected cows."""
    annotated = image_bgr.copy()
    for x, y, w, h, confidence in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"cow {confidence:.0%}"
        cv2.putText(
            annotated,
            label,
            (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 0),
            2,
        )
    return annotated
