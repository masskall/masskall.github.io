---
title: "Algorithm Reproduction: High-Performance Object Detection with YOLO-E"
date: 2025-11-15
categories:
  - Projects
tags:
  - Computer Vision
  - Deep Learning
  - PyTorch
  - Algorithm Reproduction
---

### Project Overview
As part of exploring state-of-the-art Computer Vision algorithms, I independently reproduced the **YOLO-E** (You Only Look Once) object detection architecture. The goal of this project was to deeply understand anchor-free detection mechanisms, dynamic label assignment, and to evaluate the trade-off between inference latency and mean Average Precision (mAP).

### Technical Implementation
* **Framework & Environment:** Built from scratch using Python and **PyTorch**, utilizing CUDA for GPU-accelerated training.
* **Data Processing:** Implemented custom data loaders and applied advanced data augmentation techniques (Mosaic, MixUp, and random perspective scaling) to enhance model robustness on custom datasets.
* **Model Training & Tuning:** Reconstructed the backbone and neck of YOLO-E (CSPNet, PANet). Conducted hyperparameter tuning for learning rate scheduling (Cosine Annealing) and optimized the loss function (combining GIoU loss and Distribution Focal Loss).

### Core Code Snippet (Inference Pipeline)
Here is a snippet demonstrating the custom inference pipeline and non-maximum suppression (NMS) handling:

```python
import torch
import cv2
from models.yoloe import YOLOE_Model
from utils.general import non_max_suppression

def run_inference(image_path, model, device, conf_thres=0.45, iou_thres=0.5):
    img = cv2.imread(image_path)
    img_tensor = preprocess_image(img).to(device)
    
    with torch.no_grad():
        # Forward pass
        predictions = model(img_tensor)
        
    # Apply Non-Maximum Suppression (NMS)
    outputs = non_max_suppression(predictions, conf_thres, iou_thres)
    
    return postprocess_boxes(outputs, img.shape)
