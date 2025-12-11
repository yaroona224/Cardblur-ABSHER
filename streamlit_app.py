# streamlit_app.py — CardBlur (Streamlit, Upload + Live WebRTC with anti-flicker, fast drop-off)
# - Upload image OR use true live camera (no snapshots)
# - Modes: Text only / Face only / Text + Face / Whole card
# - Live path has temporal smoothing to prevent flicker, tuned to drop blur quickly

import os, io, pathlib, urllib.request, threading
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# -------- live video deps --------
HAS_WEBRTC = False
HAS_AV = False
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase, RTCConfiguration
    HAS_WEBRTC = True
except Exception:
    pass
try:
    import av  # required by streamlit-webrtc
    HAS_AV = True
except Exception:
    pass

_infer_lock = threading.Lock()  # YOLO safety across frames

# =========================
# CONFIG (env/secrets friendly)
# =========================
BASE_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "best.pt"
WEIGHTS_PATH = pathlib.Path(os.environ.get("WEIGHTS_PATH", str(DEFAULT_WEIGHTS))).resolve()

# Inference settings
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
CONF     = float(os.environ.get("CONF", 0.28))
IOU      = float(os.environ.get("IOU", 0.5))

# Upload-only settings (tiled + flip for better recall)
UPLOAD_CONF   = float(os.environ.get("UPLOAD_CONF", 0.15))
UPLOAD_IOU    = float(os.environ.get("UPLOAD_IOU", 0.5))
TILE_SIZE     = int(os.environ.get("TILE_SIZE", 960))
TILE_OVERLAP  = float(os.environ.get("TILE_OVERLAP", 0.20))

# Live options (FAST DROP-OFF DEFAULTS)
LIVE_DETECT_EVERY   = int(os.environ.get("LIVE_DETECT_EVERY", "1"))
LIVE_ALWAYS_BLUR_DOC= os.environ.get("LIVE_ALWAYS_BLUR_DOC", "1") == "1"
LIVE_DEBUG          = os.environ.get("LIVE_DEBUG", "0") == "1"

# --- anti-flicker / smoothing (frames) ---
LIVE_HOLD_FRAMES    = int(os.environ.get("LIVE_HOLD_FRAMES", "6"))
LIVE_WARMUP_FRAMES  = int(os.environ.get("LIVE_WARMUP_FRAMES", "2"))
LIVE_MATCH_IOU      = float(os.environ.get("LIVE_MATCH_IOU", "0.35"))
LIVE_SMOOTH         = float(os.environ.get("LIVE_SMOOTH", "0.35"))

# Text post-process
TEXT_DILATE_FRAC    = float(os.environ.get("TEXT_DILATE_FRAC", 0.010))
TEXT_MERGE_GAP_FRAC = float(os.environ.get("TEXT_MERGE_GAP_FRAC", 0.010))
TEXT_MAX_DOC_FRAC   = float(os.environ.get("TEXT_MAX_DOC_FRAC", 0.50))
TEXT_MIN_H_FRAC     = float(os.environ.get("TEXT_MIN_H_FRAC", 0.012))
TEXT_MAX_H_FRAC     = float(os.environ.get("TEXT_MAX_H_FRAC", 0.28))
TEXT_MIN_AR         = float(os.environ.get("TEXT_MIN_AR", 2.3))
TEXT_MAX_AR         = float(os.environ.get("TEXT_MAX_AR", 40.0))
TEXT_NMS_IOU        = float(os.environ.get("TEXT_NMS_IOU", 0.35))

# OCR (disabled by default to keep build light)
USE_OCR       = os.environ.get("USE_OCR", "0") == "1"
OCR_LANG      = os.environ.get("OCR_LANG", "ar,en")
OCR_MIN_CONF  = float(os.environ.get("OCR_MIN_CONF", 0.60))
OCR_EXPAND_PX = int(os.environ.get("OCR_EXPAND_PX", 2))

# Blur strength
MIN_KERNEL   = int(os.environ.get("MIN_KERNEL", 31))
KERNEL_SCALE = float(os.environ.get("KERNEL_SCALE", 0.22))

# Labels (match your model)
DOC_LABELS  = {"id", "id_card", "idcard", "passport", "mrz", "serial", "number", "document", "card", "passport_id", "name", "dob", "expiry"}
FACE_LABELS = {"face", "person_face", "head"}
TEXT_LABELS = {"text"}

DOC_PAD_FRAC = float(os.environ.get("DOC_PAD_FRAC", 0.08))

# =========================
# UI
# =========================
st.set_page_config(page_title="CardBlur", page_icon="🪪", layout="wide")
st.title("🪪 CardBlur")
st.caption("AI-powered privacy protection for IDs & passports — by Shatha Khawaji • Renad Almutairi • Jury Alsultan • Yara Alsardi")

with st.sidebar:
    st.header("Options")
    blur_mode = st.radio("What to blur?", ["Text only", "Face only", "Text + Face", "Whole card"], index=3)
    st.caption("Tip: 'Whole card' will blur detected card boxes in live mode.")

# =========================
# Helpers
# =========================
def make_odd(n): return n if n % 2 == 1 else n + 1
def compute_kernel(w, h):
    k = int(max(w, h) * KERNEL_SCALE)
    k = max(k, MIN_KERNEL)
    return make_odd(k)

def blur_region(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2));     y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1: return
    k = compute_kernel(x2 - x1, y2 - y1)
    if k < 3: k = 3
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)

def center(x1, y1, x2, y2): return ((x1 + x2) // 2, (y1 + y2) // 2)
def contains(box, pt):
    x1, y1, x2, y2 = box; x, y = pt
    return (x1 <= x <= x2) and (y1 <= y <= y2)

def pad_box(box, pad_frac, W, H):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_frac), int(h * pad_frac)
    nx1 = max(0, x1 - px); ny1 = max(0, y1 - py)
    nx2 = min(W, x2 + px); ny2 = min(H, y2 + py)
    return (nx1, ny1, nx2, ny2)

def expand_px(box, px=4, py=4, W=99999, H=99999):
    x1, y1, x2, y2 = box
    return (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (ua + ub - inter + 1e-6)

def box_area(b):
    x1,y1,x2,y2=b
    return max(0,x2-x1)*max(0,y2-y1)

def merge_boxes_overlap_or_near(boxes, max_gap_px):
    if not boxes: return []
    boxes = boxes[:]
    merged = []
    used = [False]*len(boxes)

    def near_or_overlap(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        if not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1):
            return True
        h_gap = max(0, max(bx1 - ax2, ax1 - bx2))
        v_overlap = min(ay2, by2) - max(ay1, by1)
        if h_gap <= max_gap_px and v_overlap > 0: return True
        v_gap = max(0, max(by1 - ay2, ay1 - by2))
        h_overlap = min(ax2, bx2) - max(ax1, bx1)
        if v_gap <= max_gap_px and h_overlap > 0: return True
        return False

    for i in range(len(boxes)):
        if used[i]: continue
        cur = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]: continue
                if near_or_overlap(cur, boxes[j]):
                    x1 = min(cur[0], boxes[j][0]); y1 = min(cur[1], boxes[j][1])
                    x2 = max(cur[2], boxes[j][2]); y2 = max(cur[3], boxes[j][3])
                    cur = (x1,y1,x2,y2)
                    used[j] = True
                    changed = True
        merged.append(cur)
    return merged

def nms_boxes(boxes, scores=None, iou_th=0.5):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.ones(len(boxes), dtype=float) if scores is None else np.array(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep = []
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0: return 0.0
        ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        ub = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / (ua + ub - inter + 1e-6)
    while order.size > 0:
        i = int(order[0]); keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        order = np.array([k for k in rest if _iou(boxes[i], boxes[int(k)]) <= iou_th])
    return keep

# =========================
# Weights + Model
# =========================
def ensure_weights() -> str:
    for p in [BASE_DIR/"best.pt", BASE_DIR/"models"/"best.pt", WEIGHTS_PATH]:
        if p and pathlib.Path(p).exists():
            return str(pathlib.Path(p).resolve())
    url = None
    try:
        url = st.secrets.get("WEIGHTS_URL")
    except Exception:
        url = None
    if url:
        cache_dir = pathlib.Path.home()/".cache"/"cardblur"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir/"best.pt"
        if not dst.exists():
            with st.status("Downloading model weights…"):
                urllib.request.urlretrieve(url, dst)
        return str(dst.resolve())
    raise FileNotFoundError("best.pt not found. Put it in repo root or set WEIGHTS_URL in Streamlit secrets.")

@st.cache_resource(show_spinner=True)
def load_model(weights_path: str):
    from ultralytics import YOLO
    model = YOLO(weights_path)
    try:
        model.fuse()
    except Exception:
        pass
    names = getattr(model, "names", {})
    return model, names

# =========================
# Inference utilities
# =========================
def _predict(model, names, img_rgb, conf, iou):
    out = []
    try:
        results = model.predict(img_rgb, imgsz=IMG_SIZE, conf=conf, iou=iou, verbose=False)
    except Exception as e:
        st.warning(f"Inference error: {e}")
        return out
    if not results:
        return out
    res = results[0]

    def npint(x): return x.cpu().numpy().astype(int)
    def npfloat(x): return x.cpu().numpy().astype(float)

    try:
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            xyxy = npint(res.boxes.xyxy)
            cls  = res.boxes.cls.int().tolist() if hasattr(res.boxes, "cls") else [None]*len(xyxy)
            scr  = npfloat(res.boxes.conf).tolist() if hasattr(res.boxes, "conf") else [1.0]*len(xyxy)
            for coords, c, s in zip(xyxy.tolist(), cls, scr):
                label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                out.append((tuple(coords), (label or "").lower(), float(s)))
    except Exception:
        pass
    try:
        if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
            xyxy = npint(res.obb.xyxy)
            cls  = res.obb.cls.int().tolist() if hasattr(res.obb, "cls") else [None]*len(xyxy)
            scr  = npfloat(res.obb.conf).tolist() if hasattr(res.obb, "conf") else [1.0]*len(xyxy)
            for coords, c, s in zip(xyxy.tolist(), cls, scr):
                label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                out.append((tuple(coords), (label or "").lower(), float(s)))
    except Exception:
        pass
    return out

def _gather_boxes(preds):
    doc_boxes, face_boxes, face_scores, text_boxes, text_scores = [], [], [], [], []
    for (x1,y1,x2,y2), label, score in preds:
        if label in DOC_LABELS:    doc_boxes.append((x1,y1,x2,y2))
        elif label in FACE_LABELS: face_boxes.append((x1,y1,x2,y2)); face_scores.append(score)
        elif label in TEXT_LABELS: text_boxes.append((x1,y1,x2,y2)); text_scores.append(score)
    if face_boxes:
        keep = nms_boxes(face_boxes, face_scores, iou_th=0.45)
        face_boxes = [face_boxes[i] for i in keep]
    if text_boxes:
        keep = nms_boxes(text_boxes, None, iou_th=TEXT_NMS_IOU)
        text_boxes = [text_boxes[i] for i in keep]
    return doc_boxes, face_boxes, text_boxes
