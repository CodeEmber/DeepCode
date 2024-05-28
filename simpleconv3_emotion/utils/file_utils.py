'''
Author       : wyx-hhhh
Date         : 2023-04-29
LastEditTime : 2023-05-03
Description  : 处理文件的相关方法
'''
import os
from typing import List
import shutil
import random


def get_file_path(path: List[str] = [], add_sep_before=False, add_sep_affter=False) -> str:
    """获取文件路径

    Args:
        path (List[str], optional): 项目路径+文件路径. Defaults to [].
        add_sep_before (bool, optional): 是否在开头添加分隔符. Defaults to False.
        add_sep_affter (bool, optional): 是否在结尾添加分隔符. Defaults to False.

    Returns:
        str: 返回文件路径
    """
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.sep.join(path)
    all_path = os.path.join(project_path, file_path)
    if add_sep_before:
        file_path = os.sep + file_path
    if add_sep_affter:
        file_path = file_path + os.sep
    return all_path


def split_train_test(data_dir, labels, train_ratio=0.8, format='{index}.jpg'):
    """划分训练集和测试集

    Args:
        data_dir (str): 数据集路径
        labels (List): 标签列表
        train_ratio (float, optional): 分割比例. Defaults to 0.8.
        format (str, optional): 文件名格式. Defaults to '{index}.jpg'.
    """
    # 遍历文件夹下的文件，获取每个文件的路径和标签信息
    data = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            label = os.path.basename(root)
            if label not in labels:
                continue
            data.append((file_path, label))

    # 随机打乱数据集并划分训练集和测试集
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 将训练集和测试集的数据按照标签信息放置到不同的文件夹中
    for mode, dataset in [('train', train_data), ('test', test_data)]:
        for label in labels:
            label_dir = os.path.join(data_dir, mode, label)
            os.makedirs(label_dir, exist_ok=True)
            for i, (file_path, file_label) in enumerate(dataset):
                if file_label == label:
                    new_file_name = format.format(index=i, label=label)
                    shutil.copy(file_path, os.path.join(label_dir, new_file_name))


if __name__ == "__main__":
    split_train_test(get_file_path(['data', 'processed_data']), ["0", '嘟嘴'], format='{label}_{index}.jpg')
