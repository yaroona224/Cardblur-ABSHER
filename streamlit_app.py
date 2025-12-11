"""
CardBlur – Streamlit demo

Detects ID cards using a YOLOv8 model and blurs the card region
ONLY when both a face AND text are detected inside the same card.

This app supports:
- Image upload
- Live camera (via streamlit-webrtc)
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from PIL import Image

import streamlit as st

from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = BASE_DIR / "best.pt"

# These can be overridden with environment variables in Streamlit Cloud
DEFAULT_CONF = float(os.getenv("CARD_BLUR_CONF", "0.15"))  # lower = more sensitive (better for far cards)
DEFAULT_IOU = float(os.getenv("CARD_BLUR_IOU", "0.5"))
DEFAULT_IMGSZ = int(os.getenv("CARD_BLUR_IMGSZ", "960"))

# Class label groups.
# ⚠️ IMPORTANT: update these sets to match the exact class names in your trained model.
DOC_LABELS = {"id", "id_card", "idcard", "card", "passport", "document", "id_front", "id_back"}
FACE_LABELS = {"face", "person_face", "head", "id_face"}
TEXT_LABELS = {"text", "text_block", "id_text"}


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading YOLOv8 model...")
def load_model(weights_path: Path) -> YOLO:
    if not weights_path.is_file():
        st.error(f"Model weights not found at: {weights_path}")
        st.stop()
    model = YOLO(str(weights_path))
    return model


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------


def _box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _box_contains_point(box: Tuple[float, float, float, float], pt: Tuple[float, float]) -> bool:
    x1, y1, x2, y2 = box
    x, y = pt
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def _pad_box(box: Tuple[float, float, float, float], pad_frac: float, w: int, h: int) -> Tuple[int, int, int, int]:
    """Pad a box by pad_frac (e.g. 0.05 = 5%) and clamp to image extent."""
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_frac
    py = bh * pad_frac
    x1p = max(0, int(x1 - px))
    y1p = max(0, int(y1 - py))
    x2p = min(w, int(x2 + px))
    y2p = min(h, int(y2 + py))
    return x1p, y1p, x2p, y2p


# ---------------------------------------------------------------------
# Detection + blur core logic
# ---------------------------------------------------------------------


def run_yolo_inference(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
) -> Dict[str, List[Tuple[float, float, float, float]]]:
    """
    Run YOLO and return dict with doc/face/text boxes in xyxy format.
    """
    results = model.predict(image_bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    result = results[0]

    # Try to obtain names mapping
    names = getattr(model, "names", None) or getattr(model.model, "names", {})

    doc_boxes: List[Tuple[float, float, float, float]] = []
    face_boxes: List[Tuple[float, float, float, float]] = []
    text_boxes: List[Tuple[float, float, float, float]] = []

    if result.boxes is None or len(result.boxes) == 0:
        return {"doc": doc_boxes, "face": face_boxes, "text": text_boxes}

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        label = str(names.get(cls_id, cls_id))

        if label in DOC_LABELS:
            doc_boxes.append((x1, y1, x2, y2))
        elif label in FACE_LABELS:
            face_boxes.append((x1, y1, x2, y2))
        elif label in TEXT_LABELS:
            text_boxes.append((x1, y1, x2, y2))

    return {"doc": doc_boxes, "face": face_boxes, "text": text_boxes}


def choose_card_boxes_to_blur(
    boxes: Dict[str, List[Tuple[float, float, float, float]]],
    image_shape: Tuple[int, int, int],
    require_face_and_text: bool = True,
    pad_frac: float = 0.05,
) -> List[Tuple[int, int, int, int]]:
    """
    Given doc / face / text boxes, decide which card regions to blur.

    If require_face_and_text is True:
        Blur ONLY those doc boxes that contain at least one face center
        AND at least one text center.
    """
    H, W = image_shape[:2]
    doc_boxes = boxes["doc"]
    face_boxes = boxes["face"]
    text_boxes = boxes["text"]

    padded_targets: List[Tuple[int, int, int, int]] = []

    if not doc_boxes:
        return padded_targets

    face_centers = [_box_center(b) for b in face_boxes]
    text_centers = [_box_center(b) for b in text_boxes]

    for doc_box in doc_boxes:
        if require_face_and_text:
            has_face = any(_box_contains_point(doc_box, c) for c in face_centers)
            has_text = any(_box_contains_point(doc_box, c) for c in text_centers)
            if not (has_face and has_text):
                continue  # skip this card
        # Card is valid → pad a bit and add as blur target
        padded_targets.append(_pad_box(doc_box, pad_frac, W, H))

    return padded_targets


def apply_blur(
    image_bgr: np.ndarray,
    targets: List[Tuple[int, int, int, int]],
) -> np.ndarray:
    """
    Apply Gaussian blur inside each target box.
    Blur strength is automatically scaled with the box size.
    """
    if not targets:
        return image_bgr

    output = image_bgr.copy()

    for (x1, y1, x2, y2) in targets:
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(output.shape[1], int(x2))
        y2 = min(output.shape[0], int(y2))

        if x2 <= x1 or y2 <= y1:
            continue

        roi = output[y1:y2, x1:x2]

        # Blur kernel scales with region size; force odd numbers
        h, w = roi.shape[:2]
        k = max(15, int(0.25 * max(w, h)))
        if k % 2 == 0:
            k += 1

        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        output[y1:y2, x1:x2] = blurred

    return output


def process_frame(
    model: YOLO,
    frame_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
) -> np.ndarray:
    """
    Full pipeline:
    1. Run YOLO
    2. Decide which cards to blur (only those with face + text)
    3. Apply blur
    """
    boxes = run_yolo_inference(model, frame_bgr, conf=conf, iou=iou, imgsz=imgsz)
    targets = choose_card_boxes_to_blur(boxes, frame_bgr.shape, require_face_and_text=True, pad_frac=0.05)
    out = apply_blur(frame_bgr, targets)
    return out


# ---------------------------------------------------------------------
# Live camera transformer
# ---------------------------------------------------------------------


class CardBlurVideoTransformer(VideoTransformerBase):
    def __init__(self, model: YOLO, conf: float, iou: float, imgsz: int):
        super().__init__()
        self.model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        processed = process_frame(self.model, img, conf=self.conf, iou=self.iou, imgsz=self.imgsz)
        return processed


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="CardBlur – Smart ID Privacy",
        layout="wide",
    )

    st.title("🪪 CardBlur – Smart ID Privacy")
    st.write(
        "This demo automatically detects ID cards and blurs them **only when both a face and text** "
        "are detected inside the same card. Optimised for use with webcams and document cameras."
    )

    with st.sidebar:
        st.header("⚙️ Settings")

        weights_str = st.text_input(
            "Model weights path",
            value=str(DEFAULT_WEIGHTS),
            help="Path to your YOLOv8 OBB model file (best.pt).",
        )
        weights_path = Path(weights_str)

        conf = st.slider(
            "Detection confidence (lower detects smaller / further cards)",
            min_value=0.05,
            max_value=0.6,
            value=DEFAULT_CONF,
            step=0.01,
        )

        iou = st.slider(
            "NMS IoU threshold",
            min_value=0.2,
            max_value=0.8,
            value=DEFAULT_IOU,
            step=0.05,
        )

        imgsz = st.select_slider(
            "Inference image size",
            options=[640, 768, 896, 960, 1024],
            value=DEFAULT_IMGSZ,
            help="Larger sizes help detect cards from ~1m away but are slower.",
        )

        st.caption(
            "💡 Tip: For cards ~1 meter away, try **conf 0.10–0.18** and **img size 960 or 1024**."
        )

    model = load_model(weights_path)

    tab_upload, tab_live = st.tabs(["📁 Upload image", "📸 Live camera (beta)"])

    # -----------------------------
    # Upload tab
    # -----------------------------
    with tab_upload:
        st.subheader("Upload an image")
        uploaded = st.file_uploader(
            "Upload a photo containing an ID card",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded is not None:
            img = Image.open(uploaded).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            with st.spinner("Running detection and blurring…"):
                out_bgr = process_frame(model, img_bgr, conf=conf, iou=iou, imgsz=imgsz)

            out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**")
                st.image(img, use_column_width=True)
            with c2:
                st.markdown("**Blurred output**")
                st.image(out_rgb, use_column_width=True)

            # Download button
            out_pil = Image.fromarray(out_rgb)
            st.download_button(
                "⬇️ Download blurred image",
                data=_image_to_bytes(out_pil),
                file_name="cardblur_blurred.png",
                mime="image/png",
            )
        else:
            st.info("Upload a JPG/PNG image to get started.")

    # -----------------------------
    # Live camera tab
    # -----------------------------
    with tab_live:
        st.subheader("Live camera blur (beta)")

        st.write(
            "Enable your webcam to blur cards in real-time. "
            "The same rule applies: a card is blurred only when **both face and text** "
            "are detected inside it."
        )

        webrtc_streamer(
            key="cardblur-live",
            video_transformer_factory=lambda: CardBlurVideoTransformer(
                model=model,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
        )


def _image_to_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


if __name__ == "__main__":
    main()
