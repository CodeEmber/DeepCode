import os

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from evalution import calculate_average_precision, calculate_precision_recall
from model import FasterRCNNModel
from process_data import WheatProcessing
from settings import config
from utils.averager import Averager
from utils.torch_utils import set_device
from utils.logger import logger
from utils.file_utils import get_file_path

train_loader, valid_loader = WheatProcessing(config).data_processing()
device = set_device(device=config['device'])

num_classes = 2
model = FasterRCNNModel(num_classes)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
loss_hist = Averager()
epoch_num = config['epochs']

log_path = get_file_path(['results', 'save_tensorboard'])
if not os.path.exists(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(log_path)

train_itr = 0
valid_itr = 0
for epoch in range(epoch_num):
    loss_hist.reset()
    for images, targets, image_ids in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if train_itr % 5 == 0:
            logger.info(f"Iteration/train/itr #{train_itr} - Loss: {loss_value}")
            writer.add_scalar('Loss/train/itr', loss_value, train_itr)
        train_itr += 1
    logger.info(f"Iteration/train/epoch #{epoch} - Loss: {loss_hist.value}")
    writer.add_scalar('Loss/train/epoch', loss_hist.value, epoch)

    loss_hist.reset()
    for images, targets, image_ids in valid_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        model.eval()
        with torch.no_grad():
            predictions = model(images)

        precision, recall = calculate_precision_recall(predictions, targets)
        mAP = calculate_average_precision(predictions, targets, num_classes)

        logger.info(f"Iteration #{epoch} - Precision: {precision}, Recall: {recall}, mAP: {mAP}")

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        loss_hist.send(loss_value)
        valid_itr += 1
        if valid_itr % 5 == 0:
            logger.info(f"Iteration/valid/itr #{valid_itr} - Loss: {loss_value}")
            writer.add_scalar('Loss/valid/itr', loss_value, valid_itr)
    logger.info(f"Iteration/valid/epoch #{epoch} - Loss: {loss_hist.value}")
    writer.add_scalar('Loss/valid/epoch', loss_hist.value, epoch)

    if epoch % 2 == 0 or epoch == epoch_num - 1:
        torch.save(model.state_dict(), get_file_path(path=["model", f"model{epoch}.pth"]))

logger.success("Training completed successfully")
