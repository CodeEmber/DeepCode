'''
Author       : wyx-hhhh
Date         : 2024-01-18
LastEditTime : 2024-01-18
Description  : 
'''
import os
import streamlit as st
import numpy as np
import cv2
import dlib

import torch
from torchvision import datasets, models, transforms
import time
from PIL import Image
import torch.nn.functional as F

from utils.file_utils import get_file_path
# 加载模型和其他必要的库

CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
PREDICTOR_PATH = get_file_path(['data', 'shape_predictor_68_face_landmarks.dat'])
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cascade = cv2.CascadeClassifier(CASCADE_PATH)


def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3, 5)
    x, y, w, h = rects[0]
    rect = dlib.rectangle(x, y, x + w, y + h)
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])


testsize = 48  ##测试图大小
from models.model import simpleconv3

net = simpleconv3(4)  ## 定义模型
net.eval()  ## 设置推理模式，使得dropout和batchnorm等网络层在train和val模式间切换
torch.no_grad()  ## 停止autograd模块的工作，以起到加速和节省显存

## 载入模型权重
modelpath = get_file_path(["checkpoints", "model.pt"])
net.load_state_dict(torch.load(modelpath, map_location=lambda storage, loc: storage))

## 定义预处理函数
data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def a(image):
    im = np.array(image)
    try:
        rects = cascade.detectMultiScale(im, 1.3, 5)
        x, y, w, h = rects[0]
        rect = dlib.rectangle(x, y, x + w, y + h)
        landmarks = np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])
    except:
        return -1

    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0

    for i in range(48, 67):
        x = landmarks[i, 0]
        y = landmarks[i, 1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y

    roiwidth = xmax - xmin
    roiheight = ymax - ymin

    roi = im[ymin:ymax, xmin:xmax, 0:3]

    if roiwidth > roiheight:
        dstlen = 1.5 * roiwidth
    else:
        dstlen = 1.5 * roiheight

    diff_xlen = dstlen - roiwidth
    diff_ylen = dstlen - roiheight

    newx = xmin
    newy = ymin

    imagerows, imagecols, channel = im.shape
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen

    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagerows - dstlen

    roi = im[int(newy):int(newy + dstlen), int(newx):int(newx + dstlen), 0:3]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roiresized = cv2.resize(roi, (testsize, testsize))
    imgblob = data_transforms(roiresized).unsqueeze(0)
    imgblob.requires_grad = False
    predict = F.softmax(net(imgblob))
    print(predict)
    index = np.argmax(predict.detach().numpy())
    return index


# 设置页面标题
st.title("人脸表情识别系统")

# 上传图像文件
uploaded_file = st.file_uploader("请选择要识别的图像文件", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 读取上传的图像文件
    image = Image.open(uploaded_file)
    # 显示上传的图像
    st.image(image, caption='原始图像', use_column_width=True)

    # 在按钮按下时运行模型并处理图像
    if st.button("识别人物表情"):
        # 使用模型处理图像
        index = a(image)
        if index == 0:
            st.info("识别结果：无表情")
        elif index == 1:
            st.info("识别结果：撅嘴")
        elif index == 2:
            st.info("识别结果：微笑")
        elif index == 3:
            st.info("识别结果：张嘴")
        else:
            st.error("未识别到人脸")
