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
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
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
        w = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
        m = deeplabv3_resnet50(weights=w)
        m.eval()
        _model = (m, w)
    return _model


mongo = AsyncIOMotorClient(MONGO_URL)
db = mongo[DB_NAME]


@app.get("/api/health")
async def health():
    return {"status": "ok", "torch": torch.__version__, "model_loaded": _model is not None}


@app.get("/api/summary")
async def summary():
    data = dict(SUMMARY)
    data["demo_runs"] = await db.demo_runs.count_documents({})
    return data


def _segment_sync(img_bytes: bytes):
    m, w = load_model()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    W0, H0 = pil.size
    scale = 640.0 / max(W0, H0)
    if scale < 1:
        pil = pil.resize((int(W0 * scale), int(H0 * scale)), Image.BILINEAR)
    t = torchvision.transforms.functional.pil_to_tensor(pil).float() / 255.0
    t = torchvision.transforms.functional.normalize(t, mean=w.transforms().mean, std=w.transforms().std)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = m(t.unsqueeze(0))["out"][0]
    ms = (time.perf_counter() - t0) * 1000
    try:
        person = int(w.meta["categories"].index("person"))
    except Exception:
        person = 15
    mask = out.argmax(0).numpy().astype(np.uint8)
    mask = ((mask == person) * 255).astype(np.uint8)
    orig = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    overlay_bgr = orig.copy()
    blend = cv2.addWeighted(orig, 0.45, np.full_like(orig, (0, 0, 255)), 0.55, 0)
    overlay_bgr[mask > 0] = blend[mask > 0]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_bgr, contours, -1, (60, 220, 255), 2)
    ok, png = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise RuntimeError("encode failed")
    b64 = base64.b64encode(png.tobytes()).decode()
    pct = float((mask > 0).mean()) * 100
    return b64, pct, ms, [pil.width, pil.height], int(out.shape[0])


@app.post("/api/segment")
async def segment(file: UploadFile = File(...)):
    img = await file.read()
    if len(img) > 8 * 1024 * 1024:
        raise HTTPException(413, "Image larger than 8 MB")
    try:
        b64, pct, ms, size, classes = await asyncio.to_thread(_segment_sync, img)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Segmentation failed: {e}")
    await db.demo_runs.insert_one({
        "filename": file.filename,
        "person_pct": pct,
        "ms": ms,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return JSONResponse({
        "overlay_png_base64": b64,
        "person_pixel_pct": round(pct, 2),
        "inference_ms": round(ms, 1),
        "image_size": size,
        "model": MODEL_META,
        "note": "Zero-shot DeepLabV3-ResNet50 on the COCO person class: the same pretrained reference used in the study (mean per-image IoU 0.7567 on Penn-Fudan).",
    })


@app.get("/api/demo_stats")
async def demo_stats():
    n = await db.demo_runs.count_documents({})
    agg = await db.demo_runs.aggregate([
        {"$group": {"_id": None, "avg_ms": {"$avg": "$ms"}, "avg_pct": {"$avg": "$person_pct"}}}
    ]).to_list(1)
    return {
        "runs": n,
        "avg_ms": round(agg[0]["avg_ms"], 1) if agg else None,
        "avg_person_pct": round(agg[0]["avg_pct"], 2) if agg else None,
    }
