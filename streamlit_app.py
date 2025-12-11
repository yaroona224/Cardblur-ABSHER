# streamlit_app.py — CardBlur (Streamlit, Upload + Live WebRTC with anti-flicker, fast drop-off)
# - Upload image OR use true live camera (no snapshots)
# - Modes: Text only / Face only / Text + Face / Whole card
# - Live path has temporal smoothing to prevent flicker, tuned to drop blur quickly
# - Blur only applies to cards that contain BOTH text + face (ID/passport rule)

import os, io, pathlib, urllib.request, threading
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# -------- live video deps --------
HAS_WEBRTC = False
HAS_AV = False
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        VideoProcessorBase,
        RTCConfiguration,
    )
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
CONF = float(os.environ.get("CONF", 0.28))
IOU = float(os.environ.get("IOU", 0.5))

# Upload-only settings (tiled + flip for better recall)
UPLOAD_CONF = float(os.environ.get("UPLOAD_CONF", 0.15))
UPLOAD_IOU = float(os.environ.get("UPLOAD_IOU", 0.5))
TILE_SIZE = int(os.environ.get("TILE_SIZE", 960))
TILE_OVERLAP = float(os.environ.get("TILE_OVERLAP", 0.20))

# Live options (FAST DROP-OFF DEFAULTS)
LIVE_DETECT_EVERY = int(os.environ.get("LIVE_DETECT_EVERY", "1"))
LIVE_ALWAYS_BLUR_DOC = os.environ.get("LIVE_ALWAYS_BLUR_DOC", "1") == "1"
LIVE_DEBUG = os.environ.get("LIVE_DEBUG", "0") == "1"

# --- anti-flicker / smoothing (frames) ---
LIVE_HOLD_FRAMES = int(os.environ.get("LIVE_HOLD_FRAMES", "6"))
LIVE_WARMUP_FRAMES = int(os.environ.get("LIVE_WARMUP_FRAMES", "2"))
LIVE_MATCH_IOU = float(os.environ.get("LIVE_MATCH_IOU", "0.35"))
LIVE_SMOOTH = float(os.environ.get("LIVE_SMOOTH", "0.35"))

# Text post-process
TEXT_DILATE_FRAC = float(os.environ.get("TEXT_DILATE_FRAC", 0.010))
TEXT_MERGE_GAP_FRAC = float(os.environ.get("TEXT_MERGE_GAP_FRAC", 0.010))
TEXT_MAX_DOC_FRAC = float(os.environ.get("TEXT_MAX_DOC_FRAC", 0.50))
TEXT_MIN_H_FRAC = float(os.environ.get("TEXT_MIN_H_FRAC", 0.012))
TEXT_MAX_H_FRAC = float(os.environ.get("TEXT_MAX_H_FRAC", 0.28))
TEXT_MIN_AR = float(os.environ.get("TEXT_MIN_AR", 2.3))
TEXT_MAX_AR = float(os.environ.get("TEXT_MAX_AR", 40.0))
TEXT_NMS_IOU = float(os.environ.get("TEXT_NMS_IOU", 0.35))

# OCR (disabled by default to keep build light)
USE_OCR = os.environ.get("USE_OCR", "0") == "1"
OCR_LANG = os.environ.get("OCR_LANG", "ar,en")
OCR_MIN_CONF = float(os.environ.get("OCR_MIN_CONF", 0.60))
OCR_EXPAND_PX = int(os.environ.get("OCR_EXPAND_PX", 2))

# Blur strength
MIN_KERNEL = int(os.environ.get("MIN_KERNEL", 31))
KERNEL_SCALE = float(os.environ.get("KERNEL_SCALE", 0.22))

# Labels (match your model)
DOC_LABELS = {
    "id",
    "id_card",
    "idcard",
    "passport",
    "mrz",
    "serial",
    "number",
    "document",
    "card",
    "passport_id",
    "name",
    "dob",
    "expiry",
}
FACE_LABELS = {"face", "person_face", "head"}
TEXT_LABELS = {"text"}

DOC_PAD_FRAC = float(os.environ.get("DOC_PAD_FRAC", 0.08))

# =========================
# UI
# =========================
st.set_page_config(page_title="CardBlur", page_icon="🪪", layout="wide")
st.title("🪪 CardBlur")
st.caption(
    "AI-powered privacy protection for IDs & passports — by Shatha Khawaji • Renad Almutairi • Jury Alsultan • Yara Alsardi"
)

with st.sidebar:
    st.header("Options")
    blur_mode = st.radio(
        "What to blur?",
        ["Text only", "Face only", "Text + Face", "Whole card"],
        index=3,
    )
    st.caption("Tip: 'Whole card' will blur detected card boxes in live mode.")


# =========================
# Helpers
# =========================
def make_odd(n):
    return n if n % 2 == 1 else n + 1


def compute_kernel(w, h):
    k = int(max(w, h) * KERNEL_SCALE)
    k = max(k, MIN_KERNEL)
    return make_odd(k)


def blur_region(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return
    k = compute_kernel(x2 - x1, y2 - y1)
    if k < 3:
        k = 3
    img[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (k, k), 0)


def center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def contains(box, pt):
    x1, y1, x2, y2 = box
    x, y = pt
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def pad_box(box, pad_frac, W, H):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = int(w * pad_frac), int(h * pad_frac)
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(W, x2 + px)
    ny2 = min(H, y2 + py)
    return (nx1, ny1, nx2, ny2)


def expand_px(box, px=4, py=4, W=99999, H=99999):
    x1, y1, x2, y2 = box
    return (max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    ub = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (ua + ub - inter + 1e-6)


def box_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)


def merge_boxes_overlap_or_near(boxes, max_gap_px):
    if not boxes:
        return []
    boxes = boxes[:]
    merged = []
    used = [False] * len(boxes)

    def near_or_overlap(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        if not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1):
            return True
        h_gap = max(0, max(bx1 - ax2, ax1 - bx2))
        v_overlap = min(ay2, by2) - max(ay1, by1)
        if h_gap <= max_gap_px and v_overlap > 0:
            return True
        v_gap = max(0, max(by1 - ay2, ay1 - by2))
        h_overlap = min(ax2, bx2) - max(ax1, bx1)
        if v_gap <= max_gap_px and h_overlap > 0:
            return True
        return False

    for i in range(len(boxes)):
        if used[i]:
            continue
        cur = boxes[i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            for j in range(len(boxes)):
                if used[j]:
                    continue
                if near_or_overlap(cur, boxes[j]):
                    x1 = min(cur[0], boxes[j][0])
                    y1 = min(cur[1], boxes[j][1])
                    x2 = max(cur[2], boxes[j][2])
                    y2 = max(cur[3], boxes[j][3])
                    cur = (x1, y1, x2, y2)
                    used[j] = True
                    changed = True
        merged.append(cur)
    return merged


def nms_boxes(boxes, scores=None, iou_th=0.5):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.ones(len(boxes), dtype=float) if scores is None else np.array(
        scores, dtype=float
    )
    order = scores.argsort()[::-1]
    keep = []

    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        ub = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / (ua + ub - inter + 1e-6)

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = np.array(
            [k for k in rest if _iou(boxes[i], boxes[int(k)]) <= iou_th]
        )
    return keep


def enforce_card_rule(doc_boxes, face_boxes, text_boxes):
    """
    Keep only document boxes that contain BOTH:
    - at least one face box
    - at least one text box

    Also filter face_boxes and text_boxes so we only keep the ones inside
    those valid document boxes. This enforces the ID/passport rule.
    """
    if not doc_boxes:
        return [], [], []

    valid_docs = []
    for d in doc_boxes:
        has_face = any(contains(d, center(*fb)) for fb in face_boxes)
        has_text = any(contains(d, center(*tb)) for tb in text_boxes)
        if has_face and has_text:
            valid_docs.append(d)

    if not valid_docs:
        return [], [], []

    filtered_faces = [
        fb for fb in face_boxes if any(contains(d, center(*fb)) for d in valid_docs)
    ]
    filtered_texts = [
        tb for tb in text_boxes if any(contains(d, center(*tb)) for d in valid_docs)
    ]

    return valid_docs, filtered_faces, filtered_texts


# =========================
# Weights + Model
# =========================
def ensure_weights() -> str:
    for p in [BASE_DIR / "best.pt", BASE_DIR / "models" / "best.pt", WEIGHTS_PATH]:
        if p and pathlib.Path(p).exists():
            return str(pathlib.Path(p).resolve())
    url = None
    try:
        url = st.secrets.get("WEIGHTS_URL")
    except Exception:
        url = None
    if url:
        cache_dir = pathlib.Path.home() / ".cache" / "cardblur"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dst = cache_dir / "best.pt"
        if not dst.exists():
            with st.status("Downloading model weights…"):
                urllib.request.urlretrieve(url, dst)
        return str(dst.resolve())
    raise FileNotFoundError(
        "best.pt not found. Put it in repo root or set WEIGHTS_URL in Streamlit secrets."
    )


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
def _predict(model, names, img_rgb, conf, iou_th):
    out = []
    try:
        results = model.predict(
            img_rgb, imgsz=IMG_SIZE, conf=conf, iou=iou_th, verbose=False
        )
    except Exception as e:
        st.warning(f"Inference error: {e}")
        return out
    if not results:
        return out
    res = results[0]

    def npint(x):
        return x.cpu().numpy().astype(int)

    def npfloat(x):
        return x.cpu().numpy().astype(float)

    try:
        if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
            xyxy = npint(res.boxes.xyxy)
            cls = (
                res.boxes.cls.int().tolist()
                if hasattr(res.boxes, "cls")
                else [None] * len(xyxy)
            )
            scr = (
                npfloat(res.boxes.conf).tolist()
                if hasattr(res.boxes, "conf")
                else [1.0] * len(xyxy)
            )
            for coords, c, s in zip(xyxy.tolist(), cls, scr):
                label = (
                    names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                )
                out.append((tuple(coords), (label or "").lower(), float(s)))
    except Exception:
        pass
    try:
        if hasattr(res, "obb") and res.obb is not None and len(res.obb) > 0:
            xyxy = npint(res.obb.xyxy)
            cls = (
                res.obb.cls.int().tolist()
                if hasattr(res.obb, "cls")
                else [None] * len(xyxy)
            )
            scr = (
                npfloat(res.obb.conf).tolist()
                if hasattr(res.obb, "conf")
                else [1.0] * len(xyxy)
            )
            for coords, c, s in zip(xyxy.tolist(), cls, scr):
                label = (
                    names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                )
                out.append((tuple(coords), (label or "").lower(), float(s)))
    except Exception:
        pass
    return out


def _gather_boxes(preds):
    doc_boxes, face_boxes, face_scores, text_boxes, text_scores = [], [], [], [], []
    for (x1, y1, x2, y2), label, score in preds:
        if label in DOC_LABELS:
            doc_boxes.append((x1, y1, x2, y2))
        elif label in FACE_LABELS:
            face_boxes.append((x1, y1, x2, y2))
            face_scores.append(score)
        elif label in TEXT_LABELS:
            text_boxes.append((x1, y1, x2, y2))
            text_scores.append(score)

    if face_boxes:
        keep = nms_boxes(face_boxes, face_scores, iou_th=0.45)
        face_boxes = [face_boxes[i] for i in keep]
    if text_boxes:
        keep = nms_boxes(text_boxes, None, iou_th=TEXT_NMS_IOU)
        text_boxes = [text_boxes[i] for i in keep]

    # 🔒 Enforce: blur only for cards that have BOTH text + face
    doc_boxes, face_boxes, text_boxes = enforce_card_rule(
        doc_boxes, face_boxes, text_boxes
    )

    return doc_boxes, face_boxes, text_boxes


def apply_blur_from_boxes(img_bgr, doc_boxes, face_boxes, text_boxes, blur_mode):
    H, W = img_bgr.shape[:2]
    padded_docs = [pad_box(d, DOC_PAD_FRAC, W, H) for d in doc_boxes]

    if blur_mode == "Whole card":
        for d in padded_docs:
            blur_region(img_bgr, *d)
    elif blur_mode == "Face only":
        for fb in face_boxes:
            blur_region(img_bgr, *fb)
    elif blur_mode == "Text only":
        for tb in text_boxes:
            blur_region(img_bgr, *tb)
    elif blur_mode == "Text + Face":
        for fb in face_boxes:
            blur_region(img_bgr, *fb)
        for tb in text_boxes:
            blur_region(img_bgr, *tb)

    # Optional: always blur card boxes when we have them
    if LIVE_ALWAYS_BLUR_DOC and blur_mode != "Whole card":
        for d in padded_docs:
            blur_region(img_bgr, *d)


def run_inference_on_upload(img_bgr, model, names, blur_mode):
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preds = []

    # Simple tiling for large images
    if max(H, W) > TILE_SIZE:
        step = int(TILE_SIZE * (1.0 - TILE_OVERLAP))
        for y in range(0, H, step):
            for x in range(0, W, step):
                tile = img_rgb[y : y + TILE_SIZE, x : x + TILE_SIZE]
                if tile.size == 0:
                    continue
                tile_preds = _predict(
                    model, names, tile, conf=UPLOAD_CONF, iou_th=UPLOAD_IOU
                )
                for (x1, y1, x2, y2), label, score in tile_preds:
                    preds.append(
                        ((x1 + x, y1 + y, x2 + x, y2 + y), label, score)
                    )
    else:
        preds = _predict(model, names, img_rgb, conf=UPLOAD_CONF, iou_th=UPLOAD_IOU)

    doc_boxes, face_boxes, text_boxes = _gather_boxes(preds)
    apply_blur_from_boxes(img_bgr, doc_boxes, face_boxes, text_boxes, blur_mode)
    return img_bgr


# =========================
# Live video processor
# =========================
if HAS_WEBRTC:

    class LiveProcessor(VideoProcessorBase):
        def __init__(self, model, names, blur_mode):
            self.model = model
            self.names = names
            self.blur_mode = blur_mode

            self.frame_index = 0
            self.tracked_doc_boxes = []
            self.tracked_face_boxes = []
            self.tracked_text_boxes = []
            self.hold_counter = 0

        def _smooth_boxes(self, new_boxes, old_boxes):
            if not old_boxes:
                return new_boxes
            smoothed = []
            for d in new_boxes:
                best_iou = 0.0
                best_idx = -1
                for idx, od in enumerate(old_boxes):
                    v = iou(d, od)
                    if v > best_iou:
                        best_iou, best_idx = v, idx
                if best_iou >= LIVE_MATCH_IOU and best_idx >= 0:
                    od = old_boxes[best_idx]
                    s = tuple(
                        int(od[k] * LIVE_SMOOTH + d[k] * (1.0 - LIVE_SMOOTH))
                        for k in range(4)
                    )
                    smoothed.append(s)
                else:
                    smoothed.append(d)
            return smoothed

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_index += 1

            run_det = (
                self.frame_index > LIVE_WARMUP_FRAMES
                and self.frame_index % LIVE_DETECT_EVERY == 0
            )

            if run_det:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with _infer_lock:
                    preds = _predict(
                        self.model, self.names, img_rgb, conf=CONF, iou_th=IOU
                    )
                doc_boxes, face_boxes, text_boxes = _gather_boxes(preds)

                # smoothing
                doc_boxes = self._smooth_boxes(
                    doc_boxes, self.tracked_doc_boxes
                )

                self.tracked_doc_boxes = doc_boxes
                self.tracked_face_boxes = face_boxes
                self.tracked_text_boxes = text_boxes

                if doc_boxes:
                    self.hold_counter = LIVE_HOLD_FRAMES
                else:
                    self.hold_counter = max(0, self.hold_counter - 1)
            else:
                if self.hold_counter > 0:
                    self.hold_counter -= 1
                if self.hold_counter == 0:
                    self.tracked_doc_boxes = []
                    self.tracked_face_boxes = []
                    self.tracked_text_boxes = []

            doc_boxes = list(self.tracked_doc_boxes)
            face_boxes = list(self.tracked_face_boxes)
            text_boxes = list(self.tracked_text_boxes)

            apply_blur_from_boxes(img, doc_boxes, face_boxes, text_boxes, self.blur_mode)

            if LIVE_DEBUG:
                for (x1, y1, x2, y2) in doc_boxes:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

else:
    LiveProcessor = None  # dummy


# =========================
# Main app
# =========================
def main():
    # Load model once (shared for upload + live)
    weights_path = ensure_weights()
    model, names = load_model(weights_path)

    st.sidebar.write(f"Live support: HAS_WEBRTC={HAS_WEBRTC}, HAS_AV={HAS_AV}")

    tab_upload, tab_live = st.tabs(["📤 Upload image", "📹 Live camera"])

    # ---------- Upload tab ----------
    with tab_upload:
        st.subheader("Upload an image with IDs / passports")
        uploaded = st.file_uploader(
            "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
        )

        if uploaded is not None:
            bytes_data = uploaded.read()
            pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            img_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with st.spinner("Detecting and blurring…"):
                img_bgr = run_inference_on_upload(
                    img_bgr, model, names, blur_mode=blur_mode
                )

            out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            out_pil = Image.fromarray(out_rgb)

            col1, col2 = st.columns(2)
            with col1:
                st.image(pil_img, caption="Original", use_column_width=True)
            with col2:
                st.image(out_pil, caption="Blurred", use_column_width=True)

            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            st.download_button(
                "Download blurred image",
                data=buf.getvalue(),
                file_name="cardblur_blurred.png",
                mime="image/png",
            )
        else:
            st.info("Upload an image to see CardBlur in action.")

    # ---------- Live tab ----------
    with tab_live:
        st.subheader("Live camera (WebRTC)")

        if not HAS_WEBRTC or not HAS_AV:
            st.warning(
                "Live camera is unavailable because 'streamlit-webrtc' or 'av' is not installed."
            )
        else:
            st.markdown(
                "Allow the browser to use your camera. CardBlur will blur IDs/passports **only when they contain both face + text**."
            )

            webrtc_streamer(
                key="cardblur-live",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ),
                video_processor_factory=lambda: LiveProcessor(
                    model, names, blur_mode
                ),
                async_transform=True,
            )


if __name__ == "__main__":
    main()
