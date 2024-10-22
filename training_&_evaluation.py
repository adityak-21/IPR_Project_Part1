import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve
import numpy as np
from torchvision.ops import nms

class VEDAIDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith("_co.png")])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        rgb_img_path = os.path.join(self.img_dir, self.image_files[idx])
        ir_img_path = rgb_img_path.replace("_co.png", "_ir.png")
        label_path = os.path.join(self.label_dir, self.image_files[idx].replace("_co.png", ".txt"))

        rgb_img = cv2.imread(rgb_img_path)
        ir_img = cv2.imread(ir_img_path, cv2.IMREAD_GRAYSCALE)

        if rgb_img is None or ir_img is None:
            raise FileNotFoundError(f"Image file not found: {rgb_img_path} or {ir_img_path}")

        boxes = self.load_labels(label_path)

        rgb_img = cv2.resize(rgb_img, (self.img_size, self.img_size))
        ir_img = cv2.resize(ir_img, (self.img_size, self.img_size))
        ir_img = cv2.merge([ir_img, ir_img, ir_img])

        # rgb_img = rgb_img / 255.0
        # ir_img = ir_img / 255.0

        img = torch.cat([torch.tensor(rgb_img).permute(2, 0, 1), torch.tensor(ir_img).permute(2, 0, 1)], dim=0)  # [6, H, W]

        # if self.transform:
        #     img = self.transform(img)
        img = img.float()/255.0

        return img, boxes

    def load_labels(self, label_path):
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                boxes.append([float(p) for p in parts])
        return torch.tensor(boxes)
    
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

train_data = VEDAIDataset(img_dir="images-Aditya\\images\\train", label_dir="labels-Aditya\\labels\\train", img_size=640)

def collate_fn(batch):
    imgs, targets = [], []

    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1])

    imgs = torch.stack(imgs, 0)

    return imgs, targets

train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)

import torch
import torch.nn as nn

class DifferentialEnhanciveModule(nn.Module):
    def __init__(self, in_channels):
        super(DifferentialEnhanciveModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, diff_feat):
        avg_pool = self.global_avg_pool(diff_feat)
        max_pool = self.global_max_pool(diff_feat)
        attention = self.fc1(avg_pool + max_pool)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        enhanced_diff_feat = diff_feat * (1 + attention)
        return enhanced_diff_feat

class CommonSelectiveModule(nn.Module):
    def __init__(self, in_channels):
        super(CommonSelectiveModule, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, common_feat):
        attention = self.fc1(common_feat)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.softmax(attention)
        refined_common_feat = common_feat * attention
        return refined_common_feat

class CMAFF(nn.Module):
    def __init__(self, in_channels=512):
        super(CMAFF, self).__init__()
        self.differential_module = DifferentialEnhanciveModule(in_channels)
        self.common_module = CommonSelectiveModule(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, rgb_feat, ir_feat):
        diff_feat = (rgb_feat - ir_feat) / 2
        common_feat = (rgb_feat + ir_feat) / 2
        enhanced_diff_feat = self.differential_module(diff_feat)
        refined_common_feat = self.common_module(common_feat)
        fused_feat = torch.cat((enhanced_diff_feat, refined_common_feat), dim=1)
        fused_feat = self.conv1x1(fused_feat)
        return fused_feat

class YOLOv5WithCMAFF(nn.Module):
    def __init__(self, yolo_model):
        super(YOLOv5WithCMAFF, self).__init__()
        self.backbone = yolo_model.model.model.model[:10]
        self.ir_conv = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=1)
        self.ir_downsample = nn.MaxPool2d(kernel_size=32, stride=32)
        self.cmaff = CMAFF(512)

        self.head = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, img):
        rgb_img, ir_img = img[:, :3, :, :], img[:, 3:, :, :]
        rgb_feat = self.backbone(rgb_img)
        ir_feat = self.ir_conv(ir_img)
        ir_feat = self.ir_downsample(ir_feat)
        fused_feat = self.cmaff(rgb_feat, ir_feat)
        for i, layer in enumerate(self.head):
            fused_feat = layer(fused_feat)
        return fused_feat
    
# from yolov5.models.yolo import Model
# yolo_model = Model(cfg='yolov5//models//yolov5s.yaml', ch=3, nc=80)
# model = YOLOv5WithCMAFF(yolo_model)
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model = YOLOv5WithCMAFF(yolo_model).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

for param in model.backbone.parameters():
    param.requires_grad = False
import torch
import torch.optim as optim
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
classification_loss_fn = nn.BCEWithLogitsLoss()
bbox_loss_fn = nn.SmoothL1Loss()
confidence_loss_fn = nn.BCEWithLogitsLoss()

num_classes = 80
grid_size = (5, 5)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(device).float()

        optimizer.zero_grad()

        
        outputs = model(imgs)
        print(f"Model output type: {type(outputs)}, output shape: {outputs.shape}")

        class_preds = outputs[:, :num_classes, :, :]
        bbox_preds = outputs[:, num_classes:, :, :]

        bbox_coords_preds = bbox_preds[:, :4, :, :]
        conf_preds = bbox_preds[:, 4:, :, :]

        batch_classification_loss = 0.0
        batch_bbox_loss = 0.0
        batch_confidence_loss = 0.0

        for i, target in enumerate(targets):
            class_target = target[:, 0]
            bbox_target = target[:, 1:].to(device)

            bbox_target_grid = torch.zeros_like(bbox_coords_preds[i])
            class_target_grid = torch.zeros((num_classes, *grid_size)).to(device)

            for obj in target:
                class_label = int(obj[0])
                bbox = obj[1:]

                grid_x = int(bbox[0] * grid_size[0])
                grid_y = int(bbox[1] * grid_size[1])

                grid_x = min(grid_x, grid_size[0] - 1)
                grid_y = min(grid_y, grid_size[1] - 1)

                class_target_grid[class_label, grid_x, grid_y] = 1
                bbox_target_grid[:, grid_x, grid_y] = bbox


            classification_loss = classification_loss_fn(class_preds[i], class_target_grid)

            bbox_loss = bbox_loss_fn(bbox_coords_preds[i], bbox_target_grid)

            confidence_target = torch.ones_like(conf_preds[i]) if target.size(0) > 0 else torch.zeros_like(conf_preds[i])

            confidence_loss = confidence_loss_fn(conf_preds[i], confidence_target)

            print(confidence_loss)

            batch_classification_loss += classification_loss
            batch_bbox_loss += bbox_loss
            batch_confidence_loss += confidence_loss

        total_loss = batch_classification_loss + batch_bbox_loss + batch_confidence_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}]")

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

val_data = VEDAIDataset(img_dir="images-Aditya\\images\\val", label_dir="labels-Aditya\\labels\\val", img_size=640)
val_loader = DataLoader(val_data, batch_size=4, shuffle=True, collate_fn=collate_fn)

def calculate_iou(box1, box2):

    box1_x1 = (box1[0] - box1[2] / 2)
    box1_y1 = (box1[1] - box1[3] / 2)
    box1_x2 = (box1[0] + box1[2] / 2)
    box1_y2 = (box1[1] + box1[3] / 2)
    
    box2_x1 = box2[0] - box2[2] / 2
    box2_y1 = box2[1] - box2[3] / 2
    box2_x2 = box2[0] + box2[2] / 2
    box2_y2 = box2[1] + box2[3] / 2

    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    
    inter_area = max(((inter_x2 - inter_x1)*(inter_y2 - inter_y1)), 0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area

    iou = (inter_area / (union_area + 1e-6))

    return iou

from torchvision.ops import nms

def apply_nms(predictions, iou_threshold=0.2):
    boxes = predictions[:, :4]
    scores = predictions[:, 4]
    
    keep_indices = nms(boxes, scores, iou_threshold)
    
    return predictions[keep_indices]

def calculate_ap(precision, recall):
    precision = np.concatenate(([0.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))

    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def evaluate_model(model, dataloader, iou_threshold=0.1, max_boxes=4):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_positives_list = []
    scores_list = []
    num_gts = 0

    all_detections = []
    all_ground_truths = []

    with torch.no_grad():
        for imgs, targets in dataloader:
            print(f"precision = ", true_positives / (true_positives + false_positives + 1e-6))
            print(f"recall = ", true_positives / (true_positives + false_negatives + 1e-6))
            p = np.cumsum(true_positives_list) / (np.arange(len(true_positives_list)) + 1)
            r = np.cumsum(true_positives_list) / (num_gts+1)
            print(f"map = ", calculate_ap(p, r))
            imgs = imgs.to(device).float()
            outputs = model(imgs)

            class_preds = outputs[:, :num_classes, :, :]
            bbox_preds = outputs[:, num_classes:, :, :]

            for i, target in enumerate(targets):
                gt_boxes = target[:, 1:]
                pred_boxes = bbox_preds[i].detach().cpu()

                pred_scores = class_preds[i].detach().cpu().view(-1)
                pred_boxes = pred_boxes.view(-1, 4)
                
                if len(pred_scores) > 0:
                    top_indices = torch.topk(pred_scores, min(max_boxes, len(pred_scores))).indices
                    pred_boxes = pred_boxes[top_indices]

                all_ground_truths.append(gt_boxes)
                all_detections.append(pred_boxes)

                matched_gt = set()
                for pred_box in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, gt_box in enumerate(gt_boxes):
                        iou = calculate_iou(pred_box, gt_box)
                        iou_value = iou.item()

                        if iou_value > best_iou:
                            best_iou = iou_value
                            best_gt_idx = gt_idx

                    if best_iou > iou_threshold:
                        if best_gt_idx not in matched_gt:
                            true_positives += 1
                            matched_gt.add(best_gt_idx)
                            true_positives_list.append(1)
                    else:
                        false_positives += 1
                        true_positives_list.append(0)

                    scores_list.append(pred_scores[i].item())

                false_negatives += len(gt_boxes) - len(matched_gt)
                num_gts += len(gt_boxes)

    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)

    true_positives_list = np.array(true_positives_list)
    scores_list = np.array(scores_list)

    precisions = np.cumsum(true_positives_list) / (np.arange(len(true_positives_list)) + 1)
    recalls = np.cumsum(true_positives_list) / num_gts

    ap_0_5 = calculate_ap(precisions, recalls)

    return precision, recall, ap_0_5

precision, recall, map_0_5 = evaluate_model(model, val_loader)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, mAP@0.5: {map_0_5:.4f}")