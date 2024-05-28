import pandas as pd
import numpy as np
import cv2
import re
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from utils.file_utils import get_file_path


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def get_area(self, boxes):
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        return area

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        area = self.get_area(boxes)

        # there is only one class
        labels = torch.ones((records.shape[0], ), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {'image': image, 'bboxes': target['boxes'], 'labels': labels}
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


class WheatProcessing:

    def __init__(self, config):
        self.config = config
        self.train_df = pd.read_csv(get_file_path(config['train_data_path']))
        self.valid_df = pd.DataFrame()
        self.image_dir = get_file_path(config['train_image_dir'])

    def expand_bbox(self, x):
        r = np.array(re.findall(r"(\d+[.]?\d*)", x))
        if len(r) == 0:
            r = [-1, -1, -1, -1]
        return r

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def init_data(self):
        self.train_df['x'] = -1
        self.train_df['y'] = -1
        self.train_df['w'] = -1
        self.train_df['h'] = -1
        self.train_df[['x', 'y', 'w', 'h']] = np.stack(self.train_df['bbox'].apply(lambda x: self.expand_bbox(x)))
        self.train_df.drop(columns=['bbox'])
        self.train_df['x'] = self.train_df['x'].astype(np.float64)
        self.train_df['y'] = self.train_df['y'].astype(np.float64)
        self.train_df['w'] = self.train_df['w'].astype(np.float64)
        self.train_df['h'] = self.train_df['h'].astype(np.float64)
        if self.config.get('debug', False):
            self.train_df = self.train_df.sample(1000, random_state=42).reset_index(drop=True)

    def split_data(self):
        image_ids = self.train_df['image_id'].unique()
        valid_ids = image_ids[-665:]
        train_ids = image_ids[:-665]
        valid_df = self.train_df[self.train_df['image_id'].isin(valid_ids)]
        train_df = self.train_df[self.train_df['image_id'].isin(train_ids)]
        return train_df, valid_df

    def transform_data(self):
        train_transforms = A.Compose([A.Flip(0.5), ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        valid_transforms = A.Compose([ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
        return train_transforms, valid_transforms

    def get_dataset(self, transforms):
        dataset = WheatDataset(self.train_df, self.image_dir, transforms)
        return dataset

    def get_dataloader(self, dataset, batch_size=4, shuffle=False):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
        return data_loader

    def data_processing(self):
        self.init_data()
        self.train_df, self.valid_df = self.split_data()
        train_transforms, valid_transforms = self.transform_data()
        train_dataset = self.get_dataset(train_transforms)
        valid_dataset = self.get_dataset(valid_transforms)
        train_loader = self.get_dataloader(train_dataset)
        valid_loader = self.get_dataloader(valid_dataset)
        return train_loader, valid_loader
