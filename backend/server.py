import os
import io
import json
import time
import base64
import asyncio
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import cv2
import torch
import torchvision
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]

DATA_PATH = os.path.join(os.path.dirname(__file__), "data.json")
with open(DATA_PATH) as f:
    SUMMARY = json.load(f)

MODEL_META = "DeepLabV3-ResNet50, COCO-pretrained, person class (zero-shot)"

app = FastAPI(title="Person Segmentation Showcase")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def load_model():
    global _model

    if _model is None:
        weights = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        model = deeplabv3_resnet50(weights=weights)
        model.eval()
        _model = (model, weights)

    return _model


mongo = AsyncIOMotorClient(MONGO_URL)
db = mongo[DB_NAME]


@app.on_event("startup")
async def preload_model():
    # Load the model before users submit their first image.
    await asyncio.to_thread(load_model)


@app.get("/health")
async def platform_health():
    # The hosting health check requests /health.
    return {"status": "ok"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "torch": torch.__version__,
        "model_loaded": _model is not None,
    }


@app.get("/api/summary")
async def summary():
    data = dict(SUMMARY)
    data["demo_runs"] = await db.demo_runs.count_documents({})
    return data


def _segment_sync(img_bytes: bytes):
    model, weights = load_model()

    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    width, height = image.size
    scale = 640.0 / max(width, height)

    if scale < 1:
        image = image.resize(
            (int(width * scale), int(height * scale)),
            Image.BILINEAR,
        )

    tensor = torchvision.transforms.functional.pil_to_tensor(image).float() / 255.0
    tensor = torchvision.transforms.functional.normalize(
        tensor,
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    )

    start_time = time.perf_counter()

    with torch.no_grad():
        output = model(tensor.unsqueeze(0))["out"][0]

    inference_ms = (time.perf_counter() - start_time) * 1000

    try:
        person_class = int(weights.meta["categories"].index("person"))
    except Exception:
        person_class = 15

    mask = output.argmax(0).numpy().astype(np.uint8)
    mask = ((mask == person_class) * 255).astype(np.uint8)

    original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    overlay = original.copy()
    blend = cv2.addWeighted(
        original,
        0.45,
        np.full_like(original, (0, 0, 255)),
        0.55,
        0,
    )
    overlay[mask > 0] = blend[mask > 0]

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(overlay, contours, -1, (60, 220, 255), 2)

    success, png = cv2.imencode(".png", overlay)
    if not success:
        raise RuntimeError("Could not encode the segmented image")

    image_base64 = base64.b64encode(png.tobytes()).decode()
    person_percentage = float((mask > 0).mean()) * 100

    return (
        image_base64,
        person_percentage,
        inference_ms,
        [image.width, image.height],
        int(output.shape[0]),
    )


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)):
    image_bytes = await file.read()

    if len(image_bytes) > 8 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image larger than 8 MB")

    try:
        (
            image_base64,
            person_percentage,
            inference_ms,
            image_size,
            classes,
        ) = await asyncio.to_thread(_segment_sync, image_bytes)
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {error}",
        )

    await db.demo_runs.insert_one(
        {
            "filename": file.filename,
            "person_pct": person_percentage,
            "ms": inference_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    return JSONResponse(
        {
            "overlay_png_base64": image_base64,
            "person_pixel_pct": round(person_percentage, 2),
            "inference_ms": round(inference_ms, 1),
            "image_size": image_size,
            "model": MODEL_META,
            "note": (
                "Zero-shot DeepLabV3-ResNet50 on the COCO person class: "
                "the same pretrained reference used in the study "
                "(mean per-image IoU 0.7567 on Penn-Fudan)."
            ),
        }
    )


@app.get("/api/demo_stats")
async def demo_stats():
    runs = await db.demo_runs.count_documents({})
    aggregate = await db.demo_runs.aggregate(
        [
            {
                "$group": {
                    "_id": None,
                    "avg_ms": {"$avg": "$ms"},
                    "avg_pct": {"$avg": "$person_pct"},
                }
            }
        ]
    ).to_list(1)

    return {
        "runs": runs,
        "avg_ms": round(aggregate[0]["avg_ms"], 1) if aggregate else None,
        "avg_person_pct": (
            round(aggregate[0]["avg_pct"], 2) if aggregate else None
        ),
    }
