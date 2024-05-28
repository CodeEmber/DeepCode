'''
Author       : wyx-hhhh
Date         : 2023-04-29
LastEditTime : 2024-01-03
Description  : 图片预处理
'''
import os
from typing import List
import cv2
import dlib

from utils.file_utils import get_file_path

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PREDICTOR_PATH = get_file_path(['data', 'shape_predictor_68_face_landmarks.dat'])

cascade = cv2.CascadeClassifier(CASCADE_PATH)
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def image_pre_process(root_dir: str):
    """对图片进行统一处理

    Args:
        root_dir (str): 文件处理的根目录
    """
    list_dir = os.walk(root_dir)
    for root, dirs, files in list_dir:
        for file in files:
            image_path = os.path.join(root, file)
            try:
                image = cv2.imread(image_path)
                image = cv2.resize(image, (128, 128))
                cv2.imwrite(os.path.join(root_dir, file), image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            except Exception:
                os.remove(image_path)
                print('remove {}'.format(image_path))


def get_landmarks(image_path: str) -> List:
    """获取人脸关键点

    Args:
        image_path (str): 图片路径

    Returns:
        list: 人脸关键点
    """
    image = cv2.imread(image_path)
    rects = cascade.detectMultiScale(image, 1.3, 5)
    x, y, w, h = rects[0]
    rect = dlib.rectangle(x, y, x + w, y + h)
    return [(p.x, p.y) for p in predictor(image, rect).parts()]


def annotate_landmarks(image_path: str, landmarks: List):
    """标注人脸关键点

    Args:
        image_path (str): 图片路径
        landmarks (List): 人脸关键点
    """
    image = cv2.imread(image_path)
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(
            img=image,
            text=str(i),
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            fontScale=0.3,
            color=(0, 255, 0),
        )
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow('annotate_landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_lip_landmarks(image_path: str) -> List:
    """获取嘴唇关键点并进行标注

    Args:
        image_path (str): 图片路径

    Returns:
        List: 嘴唇关键点
    """
    landmarks = get_landmarks(image_path)
    image = cv2.imread(image_path)
    xmin, xmax, ymin, ymax = 1000, 0, 1000, 0
    for i in range(48, 67):
        x = landmarks[i][0]
        y = landmarks[i][1]
        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    # 将嘴唇区域扩大，使其成为一个正方形
    dstlen = max(roiwidth, roiheight) * 1.5

    # 扩大后的嘴唇区域与原嘴唇区域的差值
    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    # 扩大后的嘴唇区域的左上角坐标
    # 如果扩大后的嘴唇区域超出了图片范围，则将其移动到图片边缘
    roi_xmin = xmin - int(diff_xlen / 2) if xmin - int(diff_xlen / 2) > 0 else 0
    roi_ymin = ymin - int(diff_ylen / 2) if ymin - int(diff_ylen / 2) > 0 else 0

    roi = image[roi_ymin:roi_ymin + int(dstlen), roi_xmin:roi_xmin + int(dstlen), 0:3]
    return roi


def save_lip_landmarks(root_dir: str, save_path: str):
    """保存嘴唇关键点

    Args:
        root_dir (str): 图片文件夹路径
        save_path (str): 保存路径
    """
    list_dir = os.walk(root_dir)
    for root, dirs, files in list_dir:
        for file in files:
            image_path = os.path.join(root, file)
            try:
                roi = get_lip_landmarks(image_path)
                cv2.imwrite(os.path.join(save_path, file), roi, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            except Exception:
                os.remove(image_path)
                print('remove {}'.format(image_path))


if __name__ == '__main__':
    image_pre_process(root_dir="/Users/wyx/程序/机器学习/simpleconv3_emotion/data/processed_data/嘟嘴", )
