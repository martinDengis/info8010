# Bib Number Detection

A custom deep learning project for single-class object detection built from scratch using PyTorch. This project implements a bib number detector without relying on pretrained models.

## Project Structure

- `model.py`: Custom object detection model architecture
- `loss.py`: Loss function and evaluation metrics
- `dataset.py`: Dataset class for loading COCO-formatted data
- `dataloader.py`: Data loading utilities
- `train.py`: Training script
- `evaluate.py`: Evaluation script
- `config.yaml`: Configuration for training
- `eval_config.yaml`: Configuration for evaluation

## Features

- Custom object detection model from scratch with FPN-like architecture
- Modular and parameterizable
- Multi-scale prediction heads
- COCO format dataset loading (but data used in YOLO format at the model-level)
- Average Precision (AP) evaluation at IoU thresholds [.5, .75, .95]
- Visualization of predictions

## Dataset

The dataset is organized in COCO format with the following structure:

```
data/
  train/
    _annotations.coco.json
    [image files]
  valid/
    _annotations.coco.json
    [image files]
  test/
    _annotations.coco.json
    [image files]
```

The dataset is not included in the repository. Feel free to reach out to get access to it.

## Installation

```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python train.py --config config.yaml
```

You can modify the parameters in `config.yaml` to adjust:
- Image size
- Batch size
- Learning rate
- Number of epochs
- Network architecture
- Loss function weights

## Evaluation

The `evaluate.ipynb` Jupyter Notebook can be used to evaluate a trained model, for instance with the `eval_config.yaml` configuration file.

This will generate metrics and save visualization samples in the specified directory.

## Model Architecture

The model uses a custom architecture with:
1. A backbone network with multiple convolutional layers
2. Feature pyramid for multi-scale detection
3. Multiple prediction heads for different scales

## Loss Function

The loss function combines:
- Objectness loss
- Bounding box coordinate loss

---

🚀 **Authors**: Martin Dengis & Gilles Ooms  
📧 **Contact**: [@martinDengis](https://github.com/martinDengis) | [@giooms](https://github.com/giooms)
