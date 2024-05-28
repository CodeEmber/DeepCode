import cv2
import numpy as np
import torch

from model import FasterRCNNModel


def get_prediction(model, image, threshold):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.0
    img = torch.from_numpy(img).permute(2, 0, 1)
    img = img.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        prediction = model(img, None)
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    boxes = boxes[scores >= threshold].astype(np.int32)
    return boxes, scores


def main(uploaded_image, confidence):
    model_file_path = '/Users/wyx/程序/外包项目/WheatDetection/results/save_model/model49.pth'
    model = FasterRCNNModel(num_classes=2)
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    original_image = uploaded_image.copy()
    boxes, scores = get_prediction(model, original_image, confidence)

    for box, score in zip(boxes, scores):
        cv2.rectangle(original_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(original_image, str(round(score, 2)), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    result_dict = {}
    result_dict['original_image'] = original_image
    result_dict["boxes_num"] = len(boxes)
    result_dict["confidence"] = confidence
    return result_dict
