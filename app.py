import base64
import os

import cv2
import numpy as np
from flask import Flask, render_template, request

from cow_counter import detect_cows, draw_detections

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        upload = request.files.get("photo")
        if upload is None or upload.filename == "":
            error = "Please choose an image to upload."
        elif not allowed_file(upload.filename):
            error = "Unsupported file type. Please upload a PNG, JPG, BMP, or WEBP image."
        else:
            file_bytes = np.frombuffer(upload.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                error = "Could not read that image. Please try a different file."
            else:
                count, boxes = detect_cows(image)
                annotated = draw_detections(image, boxes)
                ok, buffer = cv2.imencode(".jpg", annotated)
                encoded_image = base64.b64encode(buffer).decode("ascii") if ok else None
                result = {"count": count, "image": encoded_image}

    return render_template("index.html", result=result, error=error)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
