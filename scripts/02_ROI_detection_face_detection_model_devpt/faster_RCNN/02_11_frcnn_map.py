import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict
from statistics import mean

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
    model = fasterrcnn_resnet50_fpn(pretrained=True)
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

def compare_predictions_with_ground_truths(formatted_predictions, ground_truths, iou_thresholds):
    stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    iou_sum = 0
    total_iou_count = 0

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

            iou_sum += best_iou
            total_iou_count += 1

            for iou_threshold in iou_thresholds:
                if best_iou > iou_threshold:
                    stats[iou_threshold]['TP'] += 1
                    matched[best_match] = True
                else:
                    stats[iou_threshold]['FP'] += 1

        for iou_threshold in iou_thresholds:
            stats[iou_threshold]['FN'] += len(true_boxes) - sum(matched)

    avg_iou = iou_sum / total_iou_count if total_iou_count > 0 else 0
    return stats, avg_iou

def compute_mAP(stats, iou_thresholds):
    mAP_sum = 0
    precisions = []
    recalls = []

    for iou_threshold in iou_thresholds:
        tp = stats[iou_threshold]['TP']
        fp = stats[iou_threshold]['FP']
        fn = stats[iou_threshold]['FN']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

        mAP_sum += precision

    mAP = mAP_sum / len(iou_thresholds)
    return mAP, precisions, recalls

def evaluate_model(model, data_loader, device, ground_truths, iou_thresholds):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating")):
            if images is None or targets is None:
                continue
            images = list(image.to(device) for image in images)

            detections = model(images)

            for idx, target in enumerate(targets):
                img_path = data_loader.dataset.image_paths[target['image_id'].item()]
                predictions[img_path] = detections[idx]

    formatted_predictions = format_predictions(predictions)
        stats, avg_iou = compare_predictions_with_ground_truths(formatted_predictions, ground_truths, iou_thresholds)

    # Compute mAP across IoU thresholds
    mAP, precisions, recalls = compute_mAP(stats, iou_thresholds)

    logger.info(f"mAP (IoU 0.5:0.95): {mAP}")
    for iou, precision, recall in zip(iou_thresholds, precisions, recalls):
        logger.info(f"IoU threshold {iou}: Precision: {precision}, Recall: {recall}")

    return mAP, avg_iou

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the test dataset and data loader
    test_dataset = FaceDataset('/path/to/test/images', transform=transform)
    logger.info(f"Test dataset size: {len(test_dataset)}")

    test_data_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # Load the trained model
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load('/storage/group/klk37/default/homebytes/code/models/final_models/frcnn_final.pth'))
    model.to(device)

    # Load ground truths
    ground_truths = get_ground_truths('/storage/group/klk37/default/homebytes/video/fbs/Faster_RCNN_face_detect/test')

    # Define IoU thresholds for mAP calculation (0.5 to 0.95)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)

    # Evaluate the model and get mAP and average IoU
    mAP, avg_iou = evaluate_model(model, test_data_loader, device, ground_truths, iou_thresholds)

    logger.info(f"Average IoU at threshold 0.5: {avg_iou}")
