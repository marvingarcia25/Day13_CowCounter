# Cow Counter

A small Flask web app that counts the number of cows in an uploaded photo
using a pretrained YOLOv4-tiny object detection model (trained on COCO,
which includes a "cow" class).

## How it works

1. Upload a photo through the web form.
2. The image is run through a YOLOv4-tiny network (via OpenCV's DNN module).
3. Detections classified as "cow" above a confidence threshold are kept
   (after non-max suppression), counted, and drawn on the image.
4. The page shows the cow count and the annotated photo.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser and upload a photo.

## Project layout

- `app.py` – Flask web server and upload handling
- `cow_counter.py` – loads the model and runs detection/drawing
- `templates/index.html` – upload form and results page
- `models/` – pretrained YOLOv4-tiny weights, config, and COCO class names
