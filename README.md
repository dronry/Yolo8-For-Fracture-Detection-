# Yolo8-For-Fracture-Detection-
The proposed model builds upon the YOLOv8-Small object detection architecture and integrates Convolutional Block Attention Modules (CBAM) to improve the model’s ability to focus on subtle and discriminative regions in X-ray images, particularly those indicative of bone fractures.

# Motivation for Attention Integration
In bone X-ray imagery, fractures often appear as small, irregular discontinuities or subtle intensity changes that can be easily overlooked by standard convolutional filters.
Conventional YOLO models may focus on dominant structures (e.g., joints, bones) instead of the fracture region.
To mitigate this, attention mechanisms are introduced to guide the model toward clinically relevant regions by enhancing salient features and suppressing background noise.

#  CBAM: Convolutional Block Attention Module

The CBAM module enhances feature representations along both channel and spatial dimensions:

# Channel Attention:
Learns what to focus on.
It computes separate channel attention maps using global average pooling and max pooling, allowing the model to emphasize informative channels (e.g., edges, textures).

# Spatial Attention:
Learns where to focus.
It aggregates spatial context using average and max pooling operations across channels, generating a spatial attention map that highlights crucial fracture regions.
Each CBAM block adaptively re-weights feature maps, leading to stronger focus on meaningful image areas.

# Integration Strategy

Instead of redesigning the YOLOv8 architecture, CBAM modules are recursively injected after convolutional layers that have output channels ≥ 128.
This ensures that attention is applied to semantically rich feature representations without increasing the computational burden in early layers.

Key integration steps:

The YOLOv8 architecture (yolov8s.yaml) is loaded without pretrained weights.
CBAM modules are then inserted into the model using a recursive injection function.
Afterwards, pretrained YOLOv8 weights (yolov8s.pt) are partially loaded with strict=False, allowing the base weights to initialize all compatible layers while skipping CBAM-specific parameters.

# Dataset Link 
https://www.kaggle.com/datasets/mahmudulhasantasin/fracatlas-original-dataset

# Improvements

By incorporating CBAM, the model:

- Enhances attention toward fine-grained fracture patterns.
- Improves feature discrimination between fractured and non-fractured regions.
- Reduces false detections from bone edges or shadows.
- Provides higher precision, recall, and mAP50 on X-ray datasets.

The YOLOv8-CBAM model integrates lightweight, adaptive attention modules into a robust detection framework, achieving improved localization and classification of fracture regions in medical X-ray images. It offers a balance between accuracy, interpretability, and computational efficiency, making it suitable for clinical-grade real-time diagnostic systems.
