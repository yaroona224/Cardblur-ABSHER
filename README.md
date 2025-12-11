# CardBlur  
**Privacy-Preserving Card & Document Blurring for Live Video and Images**

CardBlur is an AI system that detects and blurs **identity documents, faces, and sensitive text** in images and live streams.  
The project prevents accidental exposure of personal information such as ID cards and passports.

---

## 📖 Overview

Sensitive documents like **ID cards, passports, driver’s licenses, and family cards** can appear unintentionally in camera feeds.  
This creates risks such as:

- **Identity theft**  
- **Unauthorized access**  
- **Permanent data leaks**

---

## 🎯 Problem & Motivation

Accidental exposure happens in:

- Government and banking counters  
- CCTV and public spaces  
- Live streams and content creation  
- Online meetings and onboarding  

Exposed IDs can lead to:

- Identity theft  
- Fraud and impersonation  
- Privacy violations  

**CardBlur** removes sensitive information instantly and automatically.

---

## ✨ Key Features

- Detects **card regions**, **faces**, and **text regions**  
- Handles **rotated or tilted cards**  
- Works with:
  - Noisy backgrounds  
  - Low or mixed lighting  
  - Hand-held documents and motion  

---

## 🧠 Methodology

### Dataset
- *TrialforGeneratedIDs* (~3,000 images, 10 document types)  
- Additional **Saudi ID samples**  
- Custom **hand-held card** images in varied environments  

### Preprocessing
- YOLO annotation conversion  
- Bounding box verification  
- Dataset balancing & resizing  

### Model
- **YOLOv8s** detection model  
- Progressive training:
  - Base clean images  
  - Diverse backgrounds  
  - Hand-held scenarios  

---

## 🖼️ Dataset Samples

<div align="center">

<img src="https://github.com/user-attachments/assets/23803715-7971-4bbc-980f-b7d487da4df4" alt="saudi_id_06" width="45%"/>
<img src="https://github.com/user-attachments/assets/7bdf24e1-c89b-455b-8c12-0db2f5d7d867" alt="fin_id_rot_89" width="45%"/>
<img src="https://github.com/user-attachments/assets/32d60923-a238-4321-89ce-720d3398f068" alt="rus_internalpassport_rot_32" width="45%"/>
<img src="https://github.com/user-attachments/assets/760de171-8208-4cd7-96e6-520e48eacd98" alt="srb_passport_rot_47" width="45%"/>

</div>

---

## 📊 Key Results

| Training Stage | mAP50  | Train Box Loss | Val Box Loss |
|----------------|--------|----------------|--------------|
| Stage 1        | 0.9326 | 0.7288         | 0.8372       |
| Stage 2        | 0.9455 | 0.8515         | 0.9461       |
| Stage 3        | 0.9660 | 0.8587         | 0.6099       |

---

## 🚀 How It Works

1. User uploads an image or uses live camera.  
2. Frame is preprocessed for the model.  
3. YOLO detects card, face, and text regions.  
4. OCR detected text.  
5. Blur is applied to sensitive areas.  
6. Output is displayed or saved.

---

## ⚠️ Ethical Considerations

- Built for **privacy protection**, not surveillance.  
- Helps reduce exposure of personal data.  
- Must be used responsibly and lawfully.

---

## 🛠️ Installation & Usage

### Requirements
- Python 3.9+  
- Streamlit  
- OpenCV  
- PyTorch / YOLOv8  
- Model weights (e.g., `best.pt`)

### Setup
```bash
git clone https://github.com/<your-username>/CardBlur.git
cd CardBlur

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
