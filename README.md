# WF-DETR: Wavelet-Frequency DETR for Real-Time Fruit Quality Detection

This repository contains the official implementation of **WF-DETR (Wavelet-Frequency DETR)**, a novel real-time, end-to-end architecture for robust multi-fruit quality detection under challenging real-world conditions (varying illumination, diverse defect scales, and occlusion).


## ✨ Features

- **Wavelet Transform Convolution (WTConv)**: Enhances high-frequency texture features (e.g., defects) via discrete wavelet decomposition and adaptive reconstruction, improving robustness to lighting variations.
- **Parallel Perception Intra-scale Feature Interaction (PIFI)**: A dual-branch module (local detail enhancement + global context suppression) fused with spatial attention for superior multi-scale feature representation.
- **Optimized Localization**: Combines Dynamic Random Occlusion (DRO) data augmentation with the Minimum Point Distance IoU (MPDIoU) loss for precise bounding box regression on occluded and irregularly shaped fruits.
- **State-of-the-Art Performance**: Achieves **84.2% mAP50** on the custom Fruit-Quality-10 dataset, outperforming RT-DETR (+6.4%), YOLOv8m (+6.2%), and YOLOv11l (+1.8%) with efficient inference (93.8 GFLOPs).
- **Cross-Domain Generalization**: Validated on medical pill and handwritten signature datasets, demonstrating strong transferability.

## 📊 Fruit-Quality-10 Dataset

We introduce and release **Fruit-Quality-10**, a large-scale benchmark dataset for fruit quality detection:
- **34,500** high-quality images at 640×640 resolution.
- **10 fruit categories**: Apple, Banana, Orange, Pear, Grape, Pineapple, Mango, Peach, Strawberry, Watermelon.
- **Dual labels**: Each fruit is annotated as "good" (fresh) or "bad" (defective).
- **Realistic scenarios**: Includes occlusion, stacking, varying illumination, and complex backgrounds.

The dataset is publicly available: [github.com/Wenyi0829/fruit_datasets](https://github.com/Wenyi0829/fruit_datasets)

## 🏗️ Model Architecture

![WF-DETR Architecture](media/image2.png)
*Figure: Overview of the WF-DETR framework.*

Key components:
1. **Backbone**: RT-DETR-l with integrated **WTConv** modules in stage 3.
2. **Neck**: Replaces AIFI with the proposed **PIFI** module for enhanced intra-scale feature interaction.
3. **Head**: Standard DETR decoder with **MPDIoU** loss.

## 📈 Main Results

### Performance on Fruit-Quality-10

| Model        | mAP50 | mAP50-95 | Precision | Recall | Params (M) | GFLOPs |
|--------------|-------|----------|-----------|--------|------------|--------|
| YOLOv8m      | 78.0  | 58.3     | 70.1      | 73.0   | 25.9       | 78.7   |
| YOLOv11l     | 82.4  | 65.2     | 79.1      | 77.6   | 25.3       | 87.4   |
| Faster R-CNN | 79.4  | 51.4     | 78.2      | 76.3   | 41.4       | 177.2  |
| RT-DETR-l    | 77.8  | 59.6     | 79.3      | 76.0   | 32.8       | 108.1  |
| **WF-DETR**  | **84.2** | 64.6   | **84.1**  | **82.2**| 37.2       | **93.8**|

