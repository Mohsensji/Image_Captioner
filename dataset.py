# dataset.py
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch.utils.data import Dataset
import os
import json
from PIL import Image

class CocoDataset(Dataset):
    def __init__(self, img_folder, ann_file):
        super().__init__()
        self.img_folder = img_folder
        self.ann_file = ann_file

        with open(ann_file, 'r') as f:
            self.coco = json.load(f)

        self.img_id_to_ann = {ann['image_id']: ann for ann in self.coco['annotations']}
        self.images = self.coco['images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        transform = ResNet50_Weights.DEFAULT.transforms()
        # Load image
        img = Image.open(img_path).convert('RGB')
        img  = transform(img)
        # Load annotations
        ann = self.img_id_to_ann.get(img_id, [])
        
        return img, ann
