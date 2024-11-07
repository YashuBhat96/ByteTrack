# ByteTrack
### Automatic Bite Detection in Children as a part of HomeBytes project
<div style="display: flex; align-items: center;">
    <img src="assets/ByteTrack%20logo.jpg" alt="ByteTrack Logo" width="200" style="margin-right: 10px;"/>
    <img src="assets/HomeBytes%20logo.jpg" alt="HomeBytes Logo" width="200"/>
</div>


This project presents one of the first automated systems for detecting eating bites in children during meal times. Our aim is to develop a robust, high-accuracy bite detection model for use in video-recorded meal settings. We collected and analyzed in-lab meal videos featuring children aged 7-9 (n=242 videos train set; n=94 children) across four portion sizes.

## Approach

The system uses a multi-stage deep learning architecture optimized for accurate detection and classification:

### 1. **Face Detection**
   - **Primary Model**: [YOLOv7](https://github.com/WongKinYiu/yolov7), known for its speed and accuracy in object detection tasks, serves as our primary face detection model.
   - **Fallback Model**: [Faster R-CNN](https://arxiv.org/abs/1506.01497), a reliable choice for handling challenging cases where YOLOv7 might miss a detection.

### 2. **Bite Classification**
   - **Feature Extraction**: [EfficientNet](https://arxiv.org/abs/1905.11946) is employed for extracting high-level spatial features from detected face regions, ensuring compact yet expressive feature representations.
   - **Temporal Analysis**: An LSTM model processes these features over time to differentiate between bite and non-bite moments, capturing subtle temporal cues inherent in eating behavior.

## Dataset

Our dataset consists of carefully controlled in-lab videos, providing comprehensive coverage of varying portion sizes and eating styles among children. The annotated dataset supports model training and evaluation under real-world conditions with children.

## Project Structure

- **Face Detection**: YOLOv7 and Faster R-CNN configurations and weights.
- **Bite Classification**: EfficientNet for feature extraction, followed by LSTM for sequential bite classification.
- **Utilities**: Helper functions for video preprocessing, annotation, and data handling.
- **Configuration Files**: YAML files for model paths, hyperparameters, and relative paths for easy integration.

### Status ![Proof of Concept](https://img.shields.io/badge/status-proof--of--concept-blue)

### License ![License: MIT](https://img.shields.io/badge/license-MIT-green)
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


