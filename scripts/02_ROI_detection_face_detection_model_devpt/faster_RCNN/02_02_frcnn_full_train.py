import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sklearn.model_selection import KFold
import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import logging
import random
import copy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the transformations
transform = transforms.Compose([
    transforms.ToPILImage(),  # Converts numpy arrays to PIL Image objects
    transforms.ToTensor(),    # Converts PIL Image to torch.FloatTensor and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet means and stds
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

        # Log a message if target boxes are empty
        if len(target['boxes']) == 0:
            logger.warning(f"Empty target boxes for sample {idx}, image path: {img_path}")
            return None, None

        # Apply the image transformation here
        if self.transform:
            image = self.transform(image)

        return image, target

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))  # Filter out None values
    if len(batch) == 0:
        return None, None
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

def shuffle_dataset(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    return Subset(dataset, indices)

# Create the dataset instances for training
train_dataset = FaceDataset('/storage/group/klk37/default/homebytes/video/fbs/split_resized/train', transform=transform)
logger.info("Train dataset size: %d", len(train_dataset))

train_trans_dataset = FaceDataset('/storage/group/klk37/default/homebytes/video/fbs/split_resized/train_trans', transform=transform)
logger.info("Train transformed dataset size: %d", len(train_trans_dataset))

# Shuffle each dataset
train_dataset = shuffle_dataset(train_dataset)
train_trans_dataset = shuffle_dataset(train_trans_dataset)

# Combine the original and transformed/augmented datasets
combined_train_dataset = ConcatDataset([train_dataset, train_trans_dataset])

# After combining the datasets
logger.info("Combined train dataset size: %d", len(combined_train_dataset))

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def eager_outputs_patch(losses, detections):
    return losses, detections

def evaluate_loss(model, data_loader, device):
    model.train()  # Ensure model is in train mode to calculate losses
    original_eager_outputs = model.eager_outputs
    model.eager_outputs = eager_outputs_patch  # Patch the eager_outputs method

    val_loss = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if images is None or targets is None:
                continue
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict, _ = model(images, targets)  # Get both losses and detections
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            logger.info(f"Batch {batch_idx+1}/{len(data_loader)} - Validation Loss: {losses.item()}")

    model.eager_outputs = original_eager_outputs  # Restore the original eager_outputs method
    avg_val_loss = val_loss / len(data_loader)
    logger.info(f"Average Validation Loss: {avg_val_loss}")
    return avg_val_loss

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, accumulation_steps):
    model.train()
    optimizer.zero_grad()
    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}")):
        if images is None or targets is None:
            continue
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values()) / accumulation_steps
        losses.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if batch_idx % print_freq == 0:
            logger.info(f"Epoch {epoch+1} - Batch {batch_idx+1}/{len(data_loader)} - Training Loss: {losses.item() * accumulation_steps}")

def train_final_model(dataset, num_epochs, device, num_classes=2):
    # Create data loader for the whole dataset
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=16)
    
    # Initialize the model
    model = get_model(num_classes)
    model.to(device)
    
    # Define optimizer and scheduler
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    accumulation_steps = 8  # Accumulate gradients over 8 steps
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch+1}")
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, accumulation_steps=accumulation_steps)
        lr_scheduler.step()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} complete.")

    logger.info("Training on entire dataset complete.")
    torch.save(model.state_dict(), 'final_model.pth')
    logger.info("Model saved as 'final_model.pth'.")

if __name__ == '__main__':
    num_epochs = 5
    device = torch.device('cpu')

    logger.info('Starting training on entire dataset...')
    train_final_model(combined_train_dataset, num_epochs, device)
    logger.info('Training complete.')
