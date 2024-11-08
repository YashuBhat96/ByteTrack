import sys
import os
import cv2
import torch
import yaml
import numpy as np
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn as nn
from PIL import Image
import time
from tqdm import tqdm  # For progress tracking
import logging
import yaml
import csv

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add this line at the start of your script, right after the other imports
yolo_code_dir = os.path.join(script_dir, "Yolov7_custom")
yolo_code_dir = os.path.join(script_dir, "YOLOv7_custom")
sys.path.append(yolo_code_dir)

from models.experimental import attempt_load
from utils.general import non_max_suppression

# Set up logging to display in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the relative path to config.yaml from the script directory
config_path = os.path.join(script_dir, "config.yaml")

# Load configuration from YAML file
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully.")
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}. Please check the path.")
        return None

# Load configuration once and use only if valid
config = load_config(config_path)
if config is None:
    raise FileNotFoundError("Configuration file is missing. Ensure 'config.yaml' is in the script directory.")


# Load configuration once and use only if valid
config = load_config(config_path)
if config is None:
    raise FileNotFoundError("Configuration file is missing. Ensure 'config.yaml' is in the script directory.")


# Define device
device = torch.device('cpu')
logger.info(f"Using device: {device}")

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialize EfficientNet model for feature extraction
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficientnet.classifier = torch.nn.Identity()  # Remove classification layer

    def forward(self, x):
        x = self.efficientnet(x)
        return x

efficientnet = EfficientNetFeatureExtractor().to(device)
efficientnet.eval()
logger.info("Initialized EfficientNet for feature extraction")

# LSTM Model
class LSTMModelDeployment(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2, bidirectional=True):
        super(LSTMModelDeployment, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_factor, num_classes)

    def forward(self, x):
        h_lstm, _ = self.lstm(x)
        h_lstm_last = h_lstm[:, -1, :]  # Take the last output along the sequence dimension
        output = self.fc(h_lstm_last)
        return output
    

# Initialize models
def initialize_models(yolo_model_path, faster_rcnn_model_path, lstm_model_path):
    logger.info("Initializing models...")
    
    # Load YOLO model directly from weights file
    yolo_model = attempt_load(yolo_model_path, map_location=device)
    yolo_model.to(device).eval()
    logger.info(f"Loaded YOLO model from {yolo_model_path}")

    # Continue with the other models
    faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2).to(device)
    faster_rcnn_model.load_state_dict(torch.load(faster_rcnn_model_path, map_location=device))
    faster_rcnn_model.eval()
    logger.info(f"Loaded Faster R-CNN model from {faster_rcnn_model_path}")

    lstm_model = LSTMModelDeployment(input_size=1280, hidden_size=256, num_layers=3, num_classes=1, bidirectional=True).to(device)
    lstm_model.load_state_dict(torch.load(lstm_model_path, map_location=device))
    lstm_model.eval()
    logger.info(f"Loaded LSTM model from {lstm_model_path}")

    return yolo_model, faster_rcnn_model, lstm_model

# Function to create padding and masking
def pad_and_mask_features(features, max_sequence_length):
    padded_features = torch.zeros((1, max_sequence_length, features[0].size(1))).to(device)
    mask = torch.zeros((1, max_sequence_length)).to(device)

    num_features = len(features)
    for i in range(num_features):
        padded_features[0, i, :] = features[i]
        mask[0, i] = 1  # Mark this as a valid frame

    return padded_features, mask

# Dynamic weighted bounding box calculation
def dynamic_weighted_bounding_boxes(yolo_bbox, yolo_conf, frcnn_bbox, size_threshold=0.5):
    yolo_area = (yolo_bbox[2] - yolo_bbox[0]) * (yolo_bbox[3] - yolo_bbox[1])
    frcnn_area = (frcnn_bbox[2] - frcnn_bbox[0]) * (frcnn_bbox[3] - frcnn_bbox[1])
    area_ratio = yolo_area / frcnn_area if frcnn_area > 0 else 0

    yolo_weight = 1.0
    frcnn_weight = 0.0

    if yolo_conf >= 0.8:  # Strong confidence for YOLO
        yolo_weight = 1.0
        frcnn_weight = 0.0
    elif area_ratio < size_threshold:  # Smaller YOLO box, reduce YOLO weight
        yolo_weight = 0.4
        frcnn_weight = 0.6

    x1_avg = int((yolo_bbox[0] * yolo_weight + frcnn_bbox[0] * frcnn_weight))
    y1_avg = int((yolo_bbox[1] * yolo_weight + frcnn_bbox[1] * frcnn_weight))
    x2_avg = yolo_bbox[2] * yolo_weight + frcnn_bbox[2] * frcnn_weight
    y2_avg = yolo_bbox[3] * yolo_weight + frcnn_bbox[3] * frcnn_weight
    width = int(x2_avg - x1_avg)
    height = int(y2_avg - y1_avg)

    return (x1_avg, y1_avg, width, height)

# Process frame for YOLO detection
def process_frame_yolo(frame, model, target_size=(512, 320)):
    resized_frame = cv2.resize(frame, target_size)
    img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    tensor_frame = transform(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor_frame)[0]  # Extract the prediction tensor
        prediction = non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.5)
    return prediction, resized_frame

# Process frame for Faster R-CNN detection
def process_frame_frcnn(frame, model):
    tensor_frame = transform(Image.fromarray(frame)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(tensor_frame)
    return prediction

# Extract relevant parameters from config
bite_detection_params = config['bite_detection']
smooth_window_size = bite_detection_params.get('smooth_window_size', 20)  # Default to 20 if not provided
frame_interval = bite_detection_params.get('frame_interval', 30)  # Default to 30 if not provided

# Function for smoothing bite detection probabilities
def smooth_predictions(predictions, window_size):
    """
    Smooth bite detection probabilities using a moving average over a specified window size.
    """
    smoothed = np.convolve(predictions, np.ones(smooth_window_size) / smooth_window_size, mode='valid')
    return smoothed

# Function for suppressing duplicate bite detections
def suppress_duplicate_detections(predicted_bite_frames):
    """
    Suppress duplicate bite detections that are too close in time.
    Args:
        predicted_bite_frames: List of predicted frame numbers where bites were detected.
    Returns:
        A filtered list of bite frames with duplicates suppressed based on frame interval.
    """
    suppressed_bites = []
    last_bite_frame = -frame_interval  # Initialize to allow the first detection

    for bite_frame in predicted_bite_frames:
        if bite_frame - last_bite_frame >= frame_interval:
            suppressed_bites.append(bite_frame)
            last_bite_frame = bite_frame  # Update the last detected bite frame
    
    return suppressed_bites

# Extract optical flow parameters from config
optical_flow_params = config['optical_flow']
lk_win_size = optical_flow_params.get('lk_win_size', 15)  # Default to 15 if not provided
lk_max_level = optical_flow_params.get('lk_max_level', 2)  # Default to 2 if not provided
lk_criteria_eps = optical_flow_params.get('lk_criteria_eps', 0.03)  # Default to 0.03 if not provided
lk_criteria_count = optical_flow_params.get('lk_criteria_count', 10)  # Default to 10 if not provided

# Function to calculate optical flow using configurable Lucas-Kanade parameters
def calculate_lightweight_optical_flow(prev_frame, curr_frame, prev_points):
    """
    Calculate optical flow using Lucas-Kanade sparse optical flow method.
    This is a lightweight version that tracks motion for specific key points.
    """
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(lk_win_size, lk_win_size),
        maxLevel=lk_max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, lk_criteria_count, lk_criteria_eps)
    )

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow for the keypoints
    curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None, **lk_params)

    # Keep only the points with good status
    status = status.flatten()  # Ensure status is 1D
    good_new = curr_points[status == 1]
    good_old = prev_points[status == 1]

    # Calculate the average motion by computing the Euclidean distance between corresponding points
    motion_magnitude = np.linalg.norm(good_new - good_old, axis=1).mean()

    return motion_magnitude, good_new

# Extract parameters from config
validation_duration = config['processing'].get('validation_duration', 2)
motion_threshold = config['processing'].get('motion_threshold', 0.02)
corner_detection_params = config['optical_flow']
max_corners = corner_detection_params.get('maxCorners', 100)
quality_level = corner_detection_params.get('qualityLevel', 0.3)
min_distance = corner_detection_params.get('minDistance', 7)
block_size = corner_detection_params.get('blockSize', 7)

# Function to confirm bite detection using lightweight motion tracking with optical flow
def confirm_bite_with_lightweight_motion(cap, validation_duration, motion_threshold):
    """
    Confirm the bite detection using lightweight optical flow based on Lucas-Kanade method.
    Track key points and check for chewing behavior.
    """
    # Read the first frame and initialize key points to track
    success, prev_frame = cap.read()
    if not success:
        return False

    # Use Shi-Tomasi corner detection to find good features to track in the first frame
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size
    )

    # Initialize variables for tracking motion
    total_magnitude = 0
    num_valid_frames = 0

    # Process frames for the specified validation duration
    for _ in range(int(validation_duration * cap.get(cv2.CAP_PROP_FPS))):
        success, curr_frame = cap.read()
        if not success or prev_points is None:
            break

        # Calculate optical flow between frames and update key points
        motion_magnitude, prev_points = calculate_lightweight_optical_flow(prev_frame, curr_frame, prev_points)

        total_magnitude += motion_magnitude
        num_valid_frames += 1

        # Update the previous frame for the next iteration
        prev_frame = curr_frame

    # Calculate the average motion magnitude across all valid frames
    avg_motion_magnitude = total_magnitude / num_valid_frames if num_valid_frames > 0 else 0

    # Return True if the average motion magnitude exceeds the threshold (indicating chewing motion)
    return avg_motion_magnitude >= motion_threshold


# Extract output directory from config
output_dir = config['paths']['output_dir']

def save_results(video_name, bite_count, frame_number, confidence, overall_bite_rate, current_bite_rate, latency_seconds):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file = os.path.join(output_dir, f"{video_name}_results.csv")

    # Handle None values, default to 0 if they are None
    confidence = confidence if confidence is not None else 0
    overall_bite_rate = overall_bite_rate if overall_bite_rate is not None else 0
    current_bite_rate = current_bite_rate if current_bite_rate is not None else 0
    latency_seconds = latency_seconds if latency_seconds is not None else 0  # Ensure latency is not None

    # Check if the file exists
    file_exists = os.path.exists(output_file)

    # Write to CSV file without adding extra lines
    with open(output_file, 'a', newline='') as csvfile:  # `newline=''` prevents extra blank lines
        writer = csv.writer(csvfile)
        
        # Write header if the file is newly created
        if not file_exists:
            writer.writerow([
                "Bite Count", "Frame Number", "Confidence", 
                "Overall Bite Rate (bites/min)", "Current Bite Rate (bites/min)", "Latency (seconds)"
            ])
        
        # Write the data row
        writer.writerow([
            bite_count, frame_number, f"{confidence:.4f}", 
            f"{overall_bite_rate:.4f}", f"{current_bite_rate:.2f}", f"{latency_seconds:.4f}"
        ])

def log_bite(video_name, bite_count, frame_number, confidence, overall_bite_rate, current_bite_rate, output_dir, last_logged_frame, frame_interval=30):
    """
    Log bite information if the current frame is sufficiently far from the last logged bite frame.
    """
    if frame_number - last_logged_frame >= frame_interval:
        save_results(video_name, bite_count, frame_number, confidence, overall_bite_rate, current_bite_rate, output_dir)
        return frame_number  # Update the last logged frame number
    return last_logged_frame  # No change in last logged frame

def detect_and_track(video_path, yolo_model, faster_rcnn_model, lstm_model, efficientnet, config):
    logger.info(f"Processing video: {video_path}")

    current_bite_rate = 0.0
    # Extract parameters from config
    output_dir = config['paths']['output_dir']
    redetect_interval = config['processing']['redetect_interval']
    display_scale = config['processing'].get('display_scale', 0.5)
    segment_length = config['processing']['segment_length']
    bite_threshold = config['processing']['bite_threshold']
    window_step = config['processing']['window_step']
    min_frames_between_bites = config['processing']['min_frames_between_bites']
    motion_threshold = config['processing']['motion_threshold']
    validation_duration = config['processing']['validation_duration']
    show_display = config['processing'].get('show_display', False)  # Option to show display

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Initialize tracker and variables
    tracker = cv2.TrackerKCF_create()
    init_tracking = False
    frame_count = 0
    total_frames = 0
    features_accumulated = []
    bite_count = 0
    last_bbox = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(video_path).split('.')[0]
    
    if fps == 0:
        print("Error: FPS of video is 0, cannot proceed with minute-based calculations.")
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Log output file path
    output_file = os.path.join(output_dir, f"{video_name}_results.csv")
    with open(output_file, 'a') as log_file:
        writer = csv.writer(log_file)
        if os.path.getsize(output_file) == 0:
            writer.writerow([
                "Bite Count", "Frame Number", "Confidence", "Overall Bite Rate (bites/min)", "Current Bite Rate (bites/min)", "Latency (seconds)"
            ])

    # Initialize bite detection variables
    probabilities = []
    max_prob_frame = None
    max_prob_value = 0
    predicted_bite_frames = []
    current_bites = 0
    overall_bite_rate = 0

    start_time = time.time()
    previous_minute = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        total_frames += 1
        frame_capture_time = time.time()
        elapsed_time = frame_capture_time - start_time
        current_minute = int(elapsed_time // 60)

        # Update rates every minute
        if current_minute > previous_minute:
            previous_minute = current_minute
            overall_bite_rate = (bite_count / (elapsed_time / 60)) if elapsed_time > 0 else 0
            current_bite_rate = (current_bites / 1)
            current_bites = 0

        original_height, original_width = frame.shape[:2]
        display_frame = cv2.resize(frame, (int(original_width * display_scale), int(original_height * display_scale)))

        # Detection phase
        if frame_count % redetect_interval == 0 or not init_tracking:
            yolo_bbox, yolo_conf = None, 0.0
            yolo_prediction, resized_frame = process_frame_yolo(frame, yolo_model)

            if yolo_prediction and len(yolo_prediction[0]) > 0:
                element = yolo_prediction[0][0]
                yolo_bbox = element[:4].cpu().numpy()
                yolo_conf = element[4].item()
                scale_x = original_width / 512
                scale_y = original_height / 320
                yolo_bbox = [int(c * scale_x) if i % 2 == 0 else int(c * scale_y) for i, c in enumerate(yolo_bbox)]
                last_bbox = (yolo_bbox[0], yolo_bbox[1], yolo_bbox[2] - yolo_bbox[0], yolo_bbox[3] - yolo_bbox[1])

                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, last_bbox)
                init_tracking = True
                if show_display:
                    cv2.rectangle(display_frame, 
                                  (int(last_bbox[0] * display_scale), int(last_bbox[1] * display_scale)),
                                  (int((last_bbox[0] + last_bbox[2]) * display_scale), int((last_bbox[1] + last_bbox[3]) * display_scale)),
                                  (0, 255, 0), 2)

            # Fallback to Faster R-CNN if YOLO fails or confidence is low
            frcnn_bbox, frcnn_conf = None, 0.0
            if yolo_bbox is None or yolo_conf < 0.8:
                frcnn_prediction = process_frame_frcnn(frame, faster_rcnn_model)
                if len(frcnn_prediction[0]['boxes']) > 0:
                    bbox = frcnn_prediction[0]['boxes'][0].cpu().numpy()
                    frcnn_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                    frcnn_conf = frcnn_prediction[0]['scores'][0].item()
                    last_bbox = (frcnn_bbox[0], frcnn_bbox[1], frcnn_bbox[2] - frcnn_bbox[0],
                                 frcnn_bbox[3] - frcnn_bbox[1])
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, last_bbox)
                    init_tracking = True
                    if show_display:
                        cv2.rectangle(display_frame, 
                                      (int(last_bbox[0] * display_scale), int(last_bbox[1] * display_scale)),
                                      (int((last_bbox[0] + last_bbox[2]) * display_scale), int((last_bbox[1] + last_bbox[3]) * display_scale)),
                                      (255, 0, 0), 2)

                # Merge detections if both are available
                if frcnn_bbox is not None and yolo_bbox is not None:
                    last_bbox = dynamic_weighted_bounding_boxes(yolo_bbox, yolo_conf, frcnn_bbox, size_threshold=0.5)
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, last_bbox)
                    init_tracking = True
                    if show_display:
                        cv2.rectangle(display_frame, 
                                      (int(last_bbox[0] * display_scale), int(last_bbox[1] * display_scale)),
                                      (int((last_bbox[0] + last_bbox[2]) * display_scale), int((last_bbox[1] + last_bbox[3]) * display_scale)),
                                      (0, 255, 255), 2)

        # Tracking phase
        if init_tracking:
            success, tracked_bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in tracked_bbox]
                if show_display:
                    cv2.rectangle(display_frame, 
                                  (int(x * display_scale), int(y * display_scale)),
                                  (int((x + w) * display_scale), int((y + h) * display_scale)), (255, 0, 0), 2)
                roi_frame = frame[y:y + h, x:x + w]

                if roi_frame.size > 0:
                    roi_tensor = transform(Image.fromarray(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = efficientnet(roi_tensor)
                    features_accumulated.append(features)

                    if len(features_accumulated) >= segment_length:
                        padded_features, mask = pad_and_mask_features(features_accumulated[-segment_length:], segment_length)
                        with torch.no_grad():
                            outputs = lstm_model(padded_features)
                            prob = torch.sigmoid(outputs).max().item()
                        probabilities.append(prob)

                        # Track the maximum probability and the corresponding frame
                        if prob > max_prob_value:
                            max_prob_value = prob
                            max_prob_frame = frame_count

                        # Apply temporal smoothing and bite detection
                        if len(probabilities) >= window_step:
                            smoothed_probs = smooth_predictions(probabilities, window_size=20)
                            last_smoothed_prob = smoothed_probs[-1]

                            if last_smoothed_prob > bite_threshold:
                                valid_chewing = confirm_bite_with_lightweight_motion(cap, validation_duration=validation_duration, motion_threshold=motion_threshold)
                                if valid_chewing:
                                    bite_count += 1
                                    current_bites += 1
                                    detection_time = time.time()
                                    latency_seconds = detection_time - frame_capture_time
                                    # Log result
                                    with open(output_file, 'a') as log_file:
                                        log_file.write(f"{bite_count}, {frame_count}, {last_smoothed_prob:.4f}, {overall_bite_rate:.4f}, {current_bite_rate:.2f}, {latency_seconds:.4f}\n")

                            features_accumulated = features_accumulated[window_step:]
                            probabilities = probabilities[window_step:]
                            max_prob_value = 0
                            max_prob_frame = None

        # Optionally show the display
        if show_display:
            cv2.imshow("Bite Detection (Scaled)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    logger.info(f"Finished processing video: {video_path}")

# Process all videos in the specified directory
def process_videos(directory_path, yolo_model, faster_rcnn_model, lstm_model, efficientnet, config):
    output_dir = config['paths']['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    logger.info(f"Processing videos in directory: {directory_path}")
    
    # List all video files in the directory to get total count for tqdm
    video_files = [filename for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename)) and filename.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    # Wrap the processing loop with tqdm for progress tracking
    for filename in tqdm(video_files, desc="Processing Videos", unit="video"):
        video_path = os.path.join(directory_path, filename)
        detect_and_track(video_path, yolo_model, faster_rcnn_model, lstm_model, efficientnet, config)

    logger.info("All videos have been processed.")

if __name__ == "__main__":
    # Extract paths and parameters from config
    yolo_model_path = config['models']['yolo_model_path']
    faster_rcnn_model_path = config['models']['faster_rcnn_model_path']
    lstm_model_path = config['models']['lstm_model_path']
    output_dir = config['paths']['output_dir']
    video_dir = config['paths']['video_dir']
    window_size = bite_detection_params.get('window_size', 20)  # Default to 20 if not provided

    # Initialize models
    yolo_model, faster_rcnn_model, lstm_model = initialize_models(yolo_model_path, faster_rcnn_model_path, lstm_model_path)

    # Process videos in the specified directory
    process_videos(video_dir, yolo_model, faster_rcnn_model, lstm_model, efficientnet, config)