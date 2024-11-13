# ByteTrack
### Automatic Bite Detection in Children as a part of HomeBytes project
<div style="display: flex; align-items: center;">
    <img src="assets/ByteTrack%20logo.jpg" alt="ByteTrack Logo" width="300" style="margin-right: 10px;"/>
    <img src="assets/HomeBytes_logo.jpg" alt="HomeBytes Logo" width="300"/>
</div>


## Table of Contents
- [Introduction](#introduction)
- [Approach](#approach)
  - [Face Detection](#face-detection)
  - [Bite Classification](#bite-classification)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Testing and Validation Metrics](#testing-and-validation-metrics) 
- [Results](#results)
  - [Bite Detection Performance](#bite-detection-performance)
  - [Inter-Rater Reliability Assessment](#inter-rater-reliability-assessment)
- [Status](#status)
- [License](#License)
- [HomeBytes Study information](#HomeBytes-study)
- [Contact Information](#contact-info)
- [Acknowledgments](#acknowledgments)

## Introduction

This project presents one of the first automated systems for detecting eating bites in children during meal times. Our aim is to develop a robust, high-accuracy bite detection model for use in video-recorded meal settings. We collected and analyzed in-lab meal videos featuring children aged 7-9 (n=242 videos train set; n=94 children) across four portion sizes.

The videos present unique challenges, as they capture natural, often noisy behavior in children, including frequent movement, occlusions, and varied eating styles. These factors introduce significant variability in the data, making automated bite detection particularly challenging. ByteTrack aims to overcome these complexities, enabling accurate bite detection even in noisy, real-world video settings.

## Approach

The system uses a multi-stage deep learning architecture optimized for accurate detection and classification:

### 1. Face Detection
   - **Primary Model**: [YOLOv7](https://github.com/WongKinYiu/yolov7), known for its speed and accuracy in object detection tasks, serves as our primary face detection model.
   - **Fallback Model**: [Faster R-CNN](https://arxiv.org/abs/1506.01497), a reliable choice for handling challenging cases where YOLOv7 might miss a detection.

### 2. Bite Classification
   - **Feature Extraction**: [EfficientNet](https://arxiv.org/abs/1905.11946) is employed for extracting high-level spatial features from detected face regions, ensuring compact yet expressive feature representations.
   - **Temporal Analysis**: An LSTM model processes these features over time to differentiate between bite and non-bite moments, capturing subtle temporal cues inherent in eating behavior.

## Dataset

Our dataset consists of in-lab videos, where children were provided set meals with either a human reader (researcher) or audiobook. Camera angles are from the top corner (see room orientation below) providing comprehensive coverage of varying portion sizes and eating styles among children. Ground truths for the study are manually annotated timestamps using [Noldus Observer XT](https://link.springer.com/article/10.3758/BF03203406) from 2 researchers by visual observations, using a set [protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC8671353/).

<div style="display: flex; align-items: center;">
    <img src="assets/room_orient1.jpg" alt="room_orient1" width="500" style="margin-right: 30px;"/>
    <img src="assets/room_orient2.jpg" alt="room_orient2" width="500"/>
</div>

## Project Structure

- **Face Detection**: YOLOv7 and Faster R-CNN configurations and weights.
- **Bite Classification**: EfficientNet for feature extraction, followed by LSTM for sequential bite classification.
- **Utilities**: Helper functions for video preprocessing, annotation, and data handling.
- **Configuration Files**: yaml or yml files for model paths, hyperparameters, and relative paths for easy integration.

## Getting Started
## Setup
Follow these steps to set up your environment for ByteTrack.

1. Clone the ByteTrack Repository
To get started, clone the ByteTrack repository with the custom YOLOv7 submodule. This setup ensures that YOLOv7, customized for our dataset, is available in the YOLOv7_custom folder inside the ByteTrack directory.

```bash
git clone --recurse-submodules https://github.com/YashuBhat96/ByteTrack.git
cd ByteTrack
```
Our pipeline depends on YOLOv7. The customized YOLOv7 repository is available here: [YOLOv7_custom](https://github.com/YashuBhat96/Yolov7_custom_ByteTrack.git).

## 2. Set Up the Environment
Choose one of the following methods to set up the environment:

#### **A. Using the Environment File**:
For a reproducible setup, use the ByteTrack_env.yml file to create a Conda environment with all necessary dependencies.
```bash
conda env create -f ByteTrack_env.yml
conda activate ByteTrack_env
```
#### **B. Using requirements files**:

Deployment Environment: To set up an environment to use the model on your videos, install dependencies from the deployment requirements file.

```bash
pip install -r deploy_requirements.txt
````
OR 

#### **Development Environment**: To set up an environment to develop your model, install dependencies from the development requirements file.

```bash
pip install -r dev_requirements.txt
````
## 3. Configuration File (config.yml)

The `config.yml` file is essential for managing various settings in ByteTrack, such as file paths, model parameters, and other configurations needed for the bite detection pipeline.

### 1. Purpose of `config.yml`

The configuration file is used to:
- Define paths for model weights, data files, and output directories.
- Set model parameters, such as thresholds and detection parameters.
- Manage other customizable options for running the pipeline.

### 2. Installing and Setting Up `config.yml`

The repository should contain a sample configuration file, `config.yml`, which you can customize to fit your environment. To set it up:

1. Locate `config.yml` in the repository root.
2. Open the file in a text editor and modify the paths and parameters as needed. 

   For example:
   ```yml
   model_weights: "path/to/weights.pt"
   data_path: "path/to/dataset/"
   output_dir: "path/to/output/"
   threshold: 0.5
   ```
   
## Usage

Once your environment is set up and config.yml is configured, you’re ready to run ByteTrack.

### Environment Setup

1. **Activate the Environment (ByteTrack_env)**: Make sure the ByteTrack environment is active.

   ```bash
   conda activate ByteTrack_env
   ```
   
2. **Run the main script:** The main script, ByteTrack_main.py, handles the entire bite detection process.
   
Within the ByteTrack environment, open ByteTrack repo. 

If you have GPU capability:

   ```bash
    python ByteTrack_main_GPU.py --config config.yml
   ```
If you do not have GPU capability, CPU only:

   ```bash
    python ByteTrack_main_CPU.py --config config.yml
   ```

This gives you the timestamps for bites for each video in your './test' folder. 

## Testing and Validation Metrics

### ByteTrack Metrics Test

To evaluate ByteTrack’s bite detection performance on a subject-by-subject basis, we use the ByteTrack_metrics_test.py script. This script provides detailed metrics including F1 score, recall, accuracy, precision, as well as counts of True Positives (TP), False Positives (FP), and False Negatives (FN). These metrics help assess ByteTrack’s detection accuracy and highlight areas for improvement. Use the config_test.yml file to adjust settings and paths for testing.

#### Run the metrics test script to evaluate model performance

 ```bash
python ByteTrack_metrics_test.py --config config_test.yml
 ```

### ByteTrack Validation Using [LODE](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2023.1088053/full) Model

ByteTrack’s predictions are validated using the LODE model by comparing model-predicted timestamps with manually annotated ground-truth data. The ByteTrack_validation.py script (configured by config_validation.yml) calculates validation metrics such as Root Mean Square Error (RMSE) and Percent RMSE (%RMSE), providing insight into the alignment of model predictions with actual intake events.

Key Validation Metrics:

Root Mean Square Error (RMSE): Quantifies model accuracy by comparing predicted timestamps to ground truth.

Percent RMSE (%RMSE): Calculates RMSE as a percentage, standardizing comparisons across subjects.

##### Run the validation script to assess ByteTrack’s alignment with ground truth
 ```bash
python ByteTrack_validation.py --config config_validation.yml
```

### Inter-rater Reliability Assessment ([ICC(3,1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/))

The Intraclass Correlation Coefficient (ICC(3,1)) was calculated for each subject to assess inter-rater reliability in manual annotations, validating the consistency of bite annotations across annotators. Higher ICC(3,1) values indicate the level of agreement, supporting ByteTrack’s detection accuracy in real-world scenarios with varied human annotations.

We use the ICC.py script along with config_ICC.yml to configure the paths and parameters for ICC calculations.

##### Run the ICC script to calculate inter-rater reliability
 ```bash
python ICC.py --config config_ICC.yml
 ```

## Results
### Bite Detection Performance
The following metrics summarize ByteTrack’s bite detection performance across subjects:

Average Accuracy: 67.3%

Average Precision: 72.1%

Average Recall: 81.6%

Average F1 Score: 73.7%

Precision ranges from 41.7% to 99.4%, with lower precision observed in subjects where increased movement or occlusions led to higher false positives.

Recall ranges from 71.7% to 86.2%, showing ByteTrack’s ability to consistently detect true bites, with some variance due to factors such as rapid movements or occlusions.

Most subjects have consistent detection rates, with F1 scores between 56% and 84%. Subjects with greater movement or visibility challenges exhibited slightly lower precision, suggesting potential areas for further model optimization to improve robustness across varied conditions.

### Inter-Rater Reliability Assessment (ICC(3,1))
To assess inter-rater reliability in manual annotations, we calculated the Intraclass Correlation Coefficient (ICC(3,1)) for each subject. This evaluation provides insights into annotation consistency across different raters, essential for validating ByteTrack's performance in real-world scenarios.
[≥ 0.90% = excellent reliability; < 0.90, ≥ 0.75 = good reliability; < 0.75, ≥ 0.5 = moderate reliability, < 0.5 = poor reliability]

Average ICC(3,1): 0.67

Range of ICC(3,1) Values: 0.27 to 0.89

Lower ICC values (e.g., 0.27) were observed in subjects with increased movement or partial occlusions, where model may have had more difficulty in consistently labeling bites. Conversely, high ICC values (up to 0.89) indicate strong agreement among annotators, validating the reliability of annotations used in training ByteTrack.

This inter-rater reliability assessment underscores the robustness of ByteTrack’s bite detection model, especially as a pilot model, while also highlighting areas for improvement in cases with highly variable behaviors.

## Status ![Proof of Concept](https://img.shields.io/badge/status-proof--of--concept-blue)

## License ![License: MIT](https://img.shields.io/badge/license-MIT-green)
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## HomeBytes Study Information
This study serves as a proof-of-concept aimed at assessing whether eating behaviors observed in a controlled, in-lab setting can be translated to free-living conditions (at home). Our primary goal is to evaluate the validity of bite detection and intake patterns outside the lab by comparing in-lab and at-home bite rates as well as cumulative intake curves.

ByteTracker is an innovative deep-learning and computer vision-based tool designed for automatic bite detection in children. By automating the process of bite detection, ByteTracker significantly reduces labor and time costs associated with manual observations. This study focuses on:

Comparing Manual vs. AI-Based Detection: We plan to assess ByteTracker's accuracy against traditional manual annotations (gold-standard) to validate its performance.

Evaluating Lab vs. Home Conditions: We plan to explore the consistency of bite rate and intake patterns in controlled versus free-living environments, assessing ByteTracker's applicability beyond the lab.

ByteTracker represents a step forward in leveraging AI to streamline eating behavioral analysis, aiming to enhance the scalability and practicality of dietary studies in real-world settings.

## Contact Information

Contributors
This project is a collaborative effort by a team with expertise in computer vision, children's eating behavior analysis, food science, and nutrition at The Pennsylvania State University.

**Yashaswini Rajendra Bhat**  
Main Contributor/First Author, PhD Candidate, Department of Nutritional Sciences, Penn State  
Email: [ybr5070@psu.edu]  
LinkedIn: [Yashaswini Bhat](https://www.linkedin.com/in/yashubhat/)  
X: [@YashuBhat](https://x.com/YashuBhat)  

**Dr. Timothy R. Brick**  
Main Technical Advisor, Co-Principal Investigator (Co-PI), Associate Professor, Department of Human Development and Family Studies, ICDS Faculty Co-Hire, Penn State  
Email: [tbrick@psu.edu]  
RealTime Science Lab: [https://sites.psu.edu/realtimescience]  

**Dr. Kathleen L. Keller**  
Co-Principal Investigator, Department of Nutritional Sciences and Department of Food Science, Penn State  
Email: [klk37@psu.edu]  
X: [@KatKellerLab](https://x.com/katkellerlab)  
Children's Eating Behavior Lab: [https://hhd.psu.edu/nutrition/childrens-eating-lab/facility]  

**Dr. Alaina L. Pearce**  
Principal Investigator (PI), Department of Nutritional Sciences, Penn State  
Email: [azp271@psu.edu]  
X: [@AlainaPearce](https://x.com/AlainaLPearce)  
Cogneato Lab: [https://sites.psu.edu/alainapearce]  

Manual annotation contribution by Dr. Nicholas Neuwald, University at Buffalo.

## Ackowledgments
Initial funding from Institute of Computational and Data Sciences, Penn State
