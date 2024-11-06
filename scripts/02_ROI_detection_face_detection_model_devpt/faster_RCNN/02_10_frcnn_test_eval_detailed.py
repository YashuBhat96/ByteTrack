import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import logging
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_curve

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths, self.annotation_paths = self._load_image_annotation_pairs(root_dir)

    def _load_image_annotation_pairs(self, root_dir):
        image_paths = []
        annotation_paths = []
        for subdir, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.jpg'):
                    full_path = os.path.join(subdir, file)
                    annotation_path = full_path.replace('.jpg', '.xml')
                    if os.path.exists(annotation_path):
                        image_paths.append(full_path)
                        annotation_paths.append(annotation_path)
        return image_paths, annotation_paths

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"Image at path {img_path} could not be read. Skipping.")
            return None, None
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Parsing annotations
        try:
            tree = ET.parse(annotation_path)
        except ET.ParseError:
            logger.warning(f"Annotation at path {annotation_path} could not be parsed. Skipping.")
            return None, None
        root = tree.getroot()

        boxes = []
        labels = []
        for member in root.findall('object'):
            labels.append(1)  # Assuming '1' for face.

            bndbox = member.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}

        if len(target['boxes']) == 0:
            logger.warning(f"Empty target boxes for sample {idx}, image path: {img_path}")
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, target

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))
    if len(batch) == 0:
        return None, None
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def get_model(num_classes):
    # Update to use weights instead of pretrained to avoid deprecation warning
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Function to parse XML annotations and extract ground truths
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    labels = []
    for member in root.findall('object'):
        labels.append(1)  # Assuming '1' for face
        xmin = int(member.find('bndbox').find('xmin').text)
        ymin = int(member.find('bndbox').find('ymin').text)
        xmax = int(member.find('bndbox').find('xmax').text)
        ymax = int(member.find('bndbox').find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes, labels

# Function to load all ground truths from annotations
def get_ground_truths(root_dir):
    ground_truths = {}
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.xml'):
                xml_file_path = os.path.join(subdir, file)
                boxes, labels = parse_xml(xml_file_path)
                ground_truths[xml_file_path] = {'boxes': boxes, 'labels': labels}
    return ground_truths

def format_predictions(predictions, threshold=0.5):
    formatted_predictions = {}
    for img_path, prediction in predictions.items():
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        valid_indices = scores > threshold
        valid_boxes = boxes[valid_indices].tolist()
        valid_labels = labels[valid_indices].tolist()
        formatted_predictions[img_path] = (valid_boxes, valid_labels)
    return formatted_predictions

def compare_predictions_with_ground_truths(formatted_predictions, ground_truths, iou_threshold=0.5):
    stats = {'TP': 0, 'FP': 0, 'FN': 0}
    for filepath, (predicted_boxes, _) in formatted_predictions.items():
        xml_path = filepath.replace('.jpg', '.xml').replace('.png', '.xml')
        true_boxes = ground_truths.get(xml_path, {}).get('boxes', [])
        matched = [False] * len(true_boxes)
        for pred_box in predicted_boxes:
            best_iou = 0
            best_match = -1
            for i, true_box in enumerate(true_boxes):
                iou = compute_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            if best_iou > iou_threshold:
                stats['TP'] += 1
                matched[best_match] = True
            else:
                stats['FP'] += 1
        stats['FN'] += len(true_boxes) - sum(matched)
    return stats

def evaluate_model(model, data_loader, device, ground_truths, iou_threshold=0.5):
    model.eval()
    predictions = {}
    total_latency = 0
    total_confidence = 0
    count = 0
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            if images is None or targets is None:
                continue
            images = list(image.to(device) for image in images)
            start_time = time.time()
            detections = model(images)
            latency = time.time() - start_time
            total_latency += latency
            count += 1
            for idx, target in enumerate(targets):
                img_path = data_loader.dataset.image_paths[target['image_id'].item()]
                predictions[img_path] = detections[idx]
                all_scores.extend(detections[idx]['scores'].cpu().numpy())
                all_labels.extend(detections[idx]['labels'].cpu().numpy())
            total_confidence += torch.mean(detections[idx]['scores']).item()

    formatted_predictions = format_predictions(predictions)
    stats = compare_predictions_with_ground_truths(formatted_predictions, ground_truths, iou_threshold)
    avg_latency = total_latency / count
    avg_confidence = total_confidence / count
    precision = stats['TP'] / (stats['TP'] + stats['FP']) if stats['TP'] + stats['FP'] > 0 else 0
    recall = stats['TP'] / (stats['TP'] + stats['FN']) if stats['TP'] + stats['FN'] > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-Score: {f1_score}")
    logger.info(f"Stats: {stats}")
    logger.info(f"Average Latency: {avg_latency} seconds")
    logger.info(f"Average Confidence: {avg_confidence}")
    
    # PR curve calculation and saving
    precision_values, recall_values, _ = precision_recall_curve(all_labels, all_scores)
    plt.figure()
    plt.plot(recall_values, precision_values, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('/storage/group/klk37/default/homebytes/code/pr_curve.png')
    logger.info("Precision-Recall Curve saved.")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the validation dataset
    val_dataset = FaceDataset('/storage/group/klk37/default/homebytes/video/fbs/Faster_RCNN_face_detect/test', transform=transform)
    logger.info("Validation dataset size: %d", len(val_dataset))

    # Create validation data loader
    val_data_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=16)

    # Load the trained model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('/storage/group/klk37/default/homebytes/code/models/final_models/frcnn_final.pth'))
    model.to(device)

    # Load ground truths
    ground_truths = get_ground_truths('/storage/group/klk37/default/homebytes/video/fbs/Faster_RCNN_face_detect/test')

    # Evaluate the model on the validation set
    logger.info('Evaluating on validation dataset...')
    evaluate_model(model, val_data_loader, device, ground_truths, iou_threshold=0.5)
    logger.info('Evaluation complete.')
