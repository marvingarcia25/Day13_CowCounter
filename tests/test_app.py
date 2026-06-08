import io

import cv2
import numpy as np
import pytest

from app import app


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def _jpeg_bytes(width=320, height=240):
    image = np.full((height, width, 3), (60, 140, 70), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def test_get_index_shows_upload_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"<form" in response.data
    assert b"Cow Counter" in response.data


def test_post_without_file_shows_error(client):
    response = client.post("/", data={}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Please choose an image to upload." in response.data


def test_post_with_unsupported_extension_shows_error(client):
    data = {"photo": (io.BytesIO(b"not an image"), "notes.txt")}
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Unsupported file type" in response.data


def test_post_with_valid_image_shows_count_and_annotated_image(client):
    data = {"photo": (io.BytesIO(_jpeg_bytes()), "field.jpg")}
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"cows detected" in response.data
    assert b"data:image/jpeg;base64," in response.data


def test_post_with_unreadable_image_shows_error(client):
    data = {"photo": (io.BytesIO(b"\x00\x01\x02garbage"), "broken.jpg")}
    response = client.post("/", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Could not read that image" in response.data
