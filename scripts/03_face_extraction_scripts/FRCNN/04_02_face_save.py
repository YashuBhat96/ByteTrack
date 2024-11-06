import cv2
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os
import numpy as np
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_frame_timestamps(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_timestamps = {i: i / fps for i in range(frame_count)}
    cap.release()
    return frame_timestamps

def save_tracked_image(frame, tracked_bbox, output_dir, timestamp):
    try:
        if tracked_bbox is not None:
            x, y, w, h = [int(v) for v in tracked_bbox]
            roi_frame = frame[y:y+h, x:x+w]
            if roi_frame.size > 0:
                filename = f"tracked_{timestamp:.2f}.jpg"
                filepath = os.path.join(output_dir, filename)
                success = cv2.imwrite(filepath, roi_frame)
                if not success:
                    raise Exception("cv2.imwrite returned False")
    except Exception as e:
        logging.error(f"Exception occurred while saving image: {e}")

def calculate_roi(point, frame, roi_size=(510, 300), screen_width=1366, screen_height=768):
    original_x = int(point[0] * frame.shape[1] / screen_width)
    original_y = int(point[1] * frame.shape[0] / screen_height)

    x1 = max(original_x - roi_size[0] // 2, 0)
    y1 = max(original_y - roi_size[1] // 2, 0)
    x2 = min(original_x + roi_size[0] // 2, frame.shape[1])
    y2 = min(original_y + roi_size[1] // 2, frame.shape[0])

    return (x1, y1, x2, y2)

def detect_and_track(video_info):
    video_path, model, points_map, save_base_dir = video_info
    logging.info(f"Starting detection and tracking for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    tracker = None
    init_tracking = False
    frame_count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_filename = os.path.splitext(os.path.basename(video_path))[0]
    timestamps = get_frame_timestamps(video_path)

    first_frame = None
    ret, first_frame = cap.read()
    if not ret:
        logging.error(f"Failed to read first frame of {video_path}")
        cap.release()
        return
    roi = calculate_roi(points_map[base_filename], first_frame)
    video_save_dir = os.path.join(save_base_dir, base_filename)
    os.makedirs(video_save_dir, exist_ok=True)
    cap.release()

    cap = cv2.VideoCapture(video_path)
    for frame_count in tqdm(range(total_frames), desc=f"Processing {base_filename}"):
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = timestamps[frame_count]

        if frame_count % 10 == 0 or not init_tracking:
            tensor_frame = F.to_tensor(frame).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                prediction = model(tensor_frame)

            for element in prediction[0]['boxes']:
                tracked_bbox = element.numpy()
                x, y, x2, y2 = tracked_bbox
                x, y, w, h = int(x), int(y), int(x2 - x), int(y2 - y)
                if x >= 0 and y >= 0 and x+w <= frame.shape[1] and y+h <= frame.shape[0]:
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x, y, w, h))
                    init_tracking = True
                    break
            else:
                init_tracking = False

        if init_tracking and tracker is not None:
            success, tracked_bbox = tracker.update(frame)
            if success:
                save_tracked_image(frame, tracked_bbox, video_save_dir, timestamp)

    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"Finished processing video: {video_path}")

def process_videos(directory_path, model_path, save_base_dir, points_file):
    torch.set_num_threads(1)  # Avoid interference with multiprocessing

    num_classes = 2
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    points_map = {}
    with open(points_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            video_id, coords = line.split(':')
            x, y = map(int, coords.split(','))
            points_map[video_id.strip()] = (x, y)

    video_infos = []
    for filename in os.listdir(directory_path):
        base_filename = os.path.splitext(filename)[0]
        if base_filename in points_map and filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(directory_path, filename)
            video_infos.append((video_path, model, points_map, save_base_dir))

    with Pool(cpu_count() - 1) as pool:
        pool.map(detect_and_track, video_infos)

if __name__ == "__main__":
    # Example usage
    process_videos(
        "/storage/group/klk37/default/homebytes/video/fbs/PS_fixed/redo1",
        "/storage/group/klk37/default/homebytes/code/models/best_model_for_faceROI.pth",
        "/storage/group/klk37/default/homebytes/video/fbs/redone1",
        "/storage/group/klk37/default/homebytes/video/fbs/selectpoint_textfiles/PS4_points.txt"
    )

