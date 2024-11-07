# ByteTrack
### Automatic Bite Detection in Children as a part of HomeBytes project
<div style="display: flex; align-items: center;">
    <img src="assets/ByteTrack%20logo.jpg" alt="ByteTrack Logo" width="300" style="margin-right: 30px;"/>
    <img src="assets/HomeBytes%20logo.jpg" alt="HomeBytes Logo" width="300"/>
</div>


## Table of Contents
- [Introduction](#introduction)
- [Approach](#approach)
  - [Face Detection](#face-detection)
  - [Bite Classification](#bite-classification)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Status](#status)
- [License](#License)
- [HomeBytes Study information](#HomeBytes-study)
- [Contact information](#contact-info)
- [Acknowledgments](#acknowledgments)

## Introduction


This project presents one of the first automated systems for detecting eating bites in children during meal times. Our aim is to develop a robust, high-accuracy bite detection model for use in video-recorded meal settings. We collected and analyzed in-lab meal videos featuring children aged 7-9 (n=242 videos train set; n=94 children) across four portion sizes.

## Approach

The system uses a multi-stage deep learning architecture optimized for accurate detection and classification:

### 1. Face Detection
   - **Primary Model**: [YOLOv7](https://github.com/WongKinYiu/yolov7), known for its speed and accuracy in object detection tasks, serves as our primary face detection model.
   - **Fallback Model**: [Faster R-CNN](https://arxiv.org/abs/1506.01497), a reliable choice for handling challenging cases where YOLOv7 might miss a detection.

### 2. Bite Classification
   - **Feature Extraction**: [EfficientNet](https://arxiv.org/abs/1905.11946) is employed for extracting high-level spatial features from detected face regions, ensuring compact yet expressive feature representations.
   - **Temporal Analysis**: An LSTM model processes these features over time to differentiate between bite and non-bite moments, capturing subtle temporal cues inherent in eating behavior.

## Dataset

Our dataset consists of in-lab videos, where children were provided set meals with either a human reader(researcher) or audiobook. Camera angles are from the top corner providing comprehensive coverage of varying portion sizes and eating styles among children. The annotated dataset supports model training and evaluation in children's eating behavior.
<div style="display: flex; align-items: center;">
    <img src="assets/room_orient1.jpg" alt="room_orient1" width="500" style="margin-right: 30px;"/>
    <img src="assets/room_orient2.jpg" alt="room_orient2" width="500"/>
</div>

## Project Structure

- **Face Detection**: YOLOv7 and Faster R-CNN configurations and weights.
- **Bite Classification**: EfficientNet for feature extraction, followed by LSTM for sequential bite classification.
- **Utilities**: Helper functions for video preprocessing, annotation, and data handling.
- **Configuration Files**: YAML files for model paths, hyperparameters, and relative paths for easy integration.

## Getting Started
## Setup

To get started with ByteTrack, follow these steps to set up your environment:

## 1. Clone the Repository

First, clone the repository and navigate into the project directory:
This pipeline uses YOLOv7 custom trained for our dataset (https://github.com/YashuBhat96/Yolov7_custom_ByteTrack.git) which should automatically get cloned with the ByteTrack repo
```bash
git clone https://github.com/YashuBhat96/ByteTrack.git
cd ByteTrack
```
## 2. Set Up the Environment
You have two options for setting up your environment, depending on your preference and requirements.

#### **A. Using the Environment File**:
For a reproducible setup, use the ByteTrack_env.yml file to create a Conda environment with all necessary dependencies.
```bash
conda env create -f ByteTrack_env.yml
conda activate ByteTrack
```
#### **B. Using requirements files**:

Deployment Environment: To set up an environment to use the model on your videos, install dependencies from the deployment requirements file.

```bash
pip install -r deploy_requirements.txt.txt
````
OR 

#### **Development Environment**: To set up an environment to develop your model, install dependencies from the development requirements file.

```bash
pip install -r dev_requirements.txt.txt
````
## 3. Configuration File (config.yaml)

The `config.yaml` file is essential for managing various settings in ByteTrack, such as file paths, model parameters, and other configurations needed for the bite detection pipeline.

### 1. Purpose of `config.yaml`

The configuration file is used to:
- Define paths for model weights, data files, and output directories.
- Set model parameters, such as thresholds and detection parameters.
- Manage other customizable options for running the pipeline.

### 2. Installing and Setting Up `config.yaml`

The repository should contain a sample configuration file, `config.yaml`, which you can customize to fit your environment. To set it up:

1. Locate `config.yaml` in the repository root.
2. Open the file in a text editor and modify the paths and parameters as needed.

   For example:
   ```yaml
   model_weights: "path/to/weights.pt"
   data_path: "path/to/dataset/"
   output_dir: "path/to/output/"
   threshold: 0.5
   ```


## Usage

After setting up the environment and configuring `config.yaml`, you can start using ByteTrack for automatic bite detection. Ensure you have activated the environment before running the main script.

### Running the Main Script

1. **Activate the Environment (ByteTrack_env)**: Make sure the ByteTrack environment is active.

   ```bash
   conda activate ByteTrack_env
   ```
2. **Run the main script:** The main script, ByteTrack_main.py, handles the entire bite detection process. Run the following command through the terminal, specifying config.yaml as the configuration file:

```bash
python ByteTrack_main.py --config config.yaml
```

## Status ![Proof of Concept](https://img.shields.io/badge/status-proof--of--concept-blue)

## License ![License: MIT](https://img.shields.io/badge/license-MIT-green)
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


