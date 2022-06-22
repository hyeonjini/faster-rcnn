import cv2
import torch
import os
import numpy as np

import albumentations as A

from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self, annotations, data_dir, transform):

        super().__init__()
        self.data_dir = data_dir
        self.annotations = COCO(annotations)
        self.image_ids = self.annotations.getImgIds()
        self.transform = transform

    def __getitem__(self, index: int) -> None:

        # get image id
        image_id = self.image_ids[index]
        
        # load image info
        image_info = self.annotations.loadImgs(image_id)[0]
        
        # load raw image
        image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
            
        # get annotation ids
        ann_ids = self.annotations.getAnnIds(imgIds=image_info['id'])

        # load annotations info
        anns = self.annotations.loadAnns(ann_ids)
        
        # get bounding box info from annotaion
        bboxes = np.array([x['bbox'] for x in anns])

        # convert (x1, y1, w, h) to (x1, y1, x2, y2)
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        # get labels
        category_ids = np.array([x['category_id'] for x in anns])
        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)

        # transforms
        if not self.transform:            
            self.transform = A.Compose([
                ToTensorV2(p=1.0),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)

        return transformed['image'], transformed['bboxes'], transformed['category_ids']

    def __len__(self) -> int:
        return len(self.image_ids)

class COCODatasetTest(COCODataset):
    def __init__(self, annotations, data_dir, transform):
        super().__init__(annotations, data_dir, transform)

    def __getitem__(self, index: int) -> None:
        # get image id
        image_id = self.image_ids[index]

        # load image info
        image_info = self.annotations.loadImgs(image_id)[0]

        # load raw image
        image = cv2.imread(os.path.join(self.data_dir, image_info["file_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        if not self.transform:
            self.transform = A.Compose([
                ToTensorV2(p=1.0),
            ])

        transformed = self.transform(image=image)
        return transformed['image']
        
    

