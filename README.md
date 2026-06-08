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
- `Dockerfile` – container image used for deployment (runs via gunicorn)

## Deploying to Azure (Container Apps)

The app ships with a `Dockerfile`, so the quickest path is **Azure Container
Apps**, which builds the image for you from source — no local Docker or
registry setup required. Run these commands from the project root with the
[Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
(`az login` first):

```bash
# 1. Pick names/region (edit as you like)
RESOURCE_GROUP=cow-counter-rg
LOCATION=eastus
ENV_NAME=cow-counter-env
APP_NAME=cow-counter

# 2. Create a resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# 3. Make sure the Container Apps extension/provider is ready (one-time)
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights

# 4. Create the Container Apps environment
az containerapp env create \
  --name $ENV_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# 5. Build the image from source and deploy it in one step
az containerapp up \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --environment $ENV_NAME \
  --source . \
  --target-port 8000 \
  --ingress external
```

When it finishes, `az containerapp up` prints a public URL
(`https://cow-counter.<random>.<region>.azurecontainerapps.io`) — open it in
your browser and try uploading a photo.

### Alternative: Azure App Service for Containers

If you'd rather use App Service:

```bash
az webapp up \
  --name $APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku B1 \
  --runtime "PYTHON:3.11"
```

Azure will detect the `Dockerfile`/`requirements.txt` and build + deploy the
app automatically. Note that the model files in `models/` total ~25 MB, and
the image needs the `libgl1`/`libglib2.0-0` system packages that the
`Dockerfile` installs for OpenCV — using the Dockerfile-based deploy
(`az containerapp up` or `az webapp up` with a container runtime) is the most
reliable route.

### Cleaning up

To avoid ongoing charges, delete the resource group when you're done:

```bash
az group delete --name $RESOURCE_GROUP --yes --no-wait
```
