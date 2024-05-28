# 导入必要的库
import torch
import numpy as np
from torchvision.ops import box_iou


# 定义函数计算精度和召回率
def calculate_precision_recall(predictions, targets, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for pred, target in zip(predictions, targets):
        iou_matrix = box_iou(pred['boxes'], target['boxes'])
        max_iou, _ = iou_matrix.max(dim=1)

        true_positives += (max_iou >= iou_threshold).sum().item()
        false_positives += (max_iou < iou_threshold).sum().item()
        false_negatives += len(target['boxes']) - (max_iou >= iou_threshold).sum().item()

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def calculate_average_precision(predictions, targets, num_classes, iou_threshold=0.5):
    all_precision = []
    all_recall = []
    for class_idx in range(num_classes):
        class_preds = [pred for pred in predictions if pred['labels'] == class_idx]
        class_targets = [target for target in targets if target['labels'] == class_idx]

        if not class_preds or not class_targets:
            continue

        iou_matrix = box_iou(torch.cat([pred['boxes'] for pred in class_preds]), torch.cat([target['boxes'] for target in class_targets]))

        for pred, target in zip(class_preds, class_targets):
            iou_scores = iou_matrix[0]
            max_iou, max_idx = iou_scores.max(dim=0)

            if max_iou.item() > iou_threshold:
                # True positive
                all_precision.append(1)
                all_recall.append(1)
            else:
                # False positive
                all_precision.append(0)
                all_recall.append(0)

    precision = sum(all_precision) / len(all_precision) if all_precision else 0
    recall = sum(all_recall) / len(all_recall) if all_recall else 0
    mAP = precision * recall  # This is a simple calculation, you may need to use a more sophisticated method for mAP

    return mAP
