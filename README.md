# Automated Bib Number Recognition for Race Events

For our Deep Learning class project, we aim to develop a **computer vision** system for recognizing and classifying bib numbers in race event photos. The system will automate the tagging process for photographers, allowing participants to quickly find their pictures.

## Features

- **Bib Detection**: Object detection model (e.g., YOLO, Faster R-CNN, DETR) to locate bibs in images.
- **Number Recognition**: OCR-based system (e.g., CRNN, TrOCR) to extract bib numbers.
- **Optional Enhancements**:
  - Basic tagging system for automatic photo labeling.
  - Confidence scoring for manual review.
  - Fallback strategies, such as face recognition.

## Dataset

Our dataset consists of **~2,500 images** from local race events. Images were scraped from the web and feature:
- Varying lighting, weather, and occlusions.
- Different bib styles, fonts, and sizes.
- Mixed-quality images (e.g., motion blur, low resolution).

## Technical Approach

We are considering:
1. A **two-step approach** (e.g., YOLO for detection + TrOCR for recognition).
2. An **end-to-end approach** using transformer-based models (e.g., Pix2Seq).

## Related Work

- CNN-based methods for text detection.
- CRNNs for sequential number recognition.
- Transformer-based models (e.g., TrOCR, Pix2Seq) for end-to-end recognition.

---

🚀 **Authors**: Martin Dengis & Gilles Ooms  
📧 **Contact**: @martinDengis | @giooms
