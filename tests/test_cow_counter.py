import numpy as np

from cow_counter import COW_CLASS_ID, detect_cows, draw_detections


def _green_field(width=320, height=240):
    return np.full((height, width, 3), (60, 140, 70), dtype=np.uint8)


def test_cow_class_id_resolves_to_cow_in_coco_names():
    assert COW_CLASS_ID == 19


def test_detect_cows_returns_no_boxes_on_an_empty_field():
    count, boxes = detect_cows(_green_field())
    assert count == 0
    assert boxes == []


def test_detect_cows_return_shape_is_consistent():
    count, boxes = detect_cows(_green_field())
    assert count == len(boxes)
    for box in boxes:
        x, y, w, h, confidence = box
        assert isinstance(x, (int, np.integer))
        assert isinstance(y, (int, np.integer))
        assert isinstance(w, (int, np.integer))
        assert isinstance(h, (int, np.integer))
        assert 0.0 <= confidence <= 1.0


def test_draw_detections_returns_same_shape_image_without_mutating_input():
    image = _green_field()
    original = image.copy()
    annotated = draw_detections(image, [(10, 10, 50, 30, 0.75)])

    assert annotated.shape == image.shape
    assert np.array_equal(image, original)  # input must not be mutated
    assert not np.array_equal(annotated, image)  # box should have been drawn
