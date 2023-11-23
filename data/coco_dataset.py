import os
import json
import numpy as np
from PIL import Image

class COCOBboxDataset:
    def __init__(self, data_dir, split='train2017'):
        self.data_dir = data_dir
        self.split = split

        with open(os.path.join(data_dir, 'annotations', f'instances_{split}.json'), 'r') as f:
            self.data = json.load(f)

        self.images = {image['id']: image for image in self.data['images']}

        self.annotations = {anno['image_id']: [] for anno in self.data['annotations']}
        for anno in self.data['annotations']:
            self.annotations[anno['image_id']].append(anno)

        self.label_map = self.create_label_map(self.data['categories'])

        self.ids = list(self.images.keys())

    def create_label_map(self, categories):
        label_map = {category['id']: idx for idx, category in enumerate(categories)}
        return label_map

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        image_id = self.ids[i]
        image_info = self.images[image_id]
        annotations = self.annotations.get(image_id, [])

        img_path = os.path.join(self.data_dir, self.split, image_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img).astype(np.float32)
        img = img.transpose(2, 0, 1)

        bboxes, labels, difficults = [], [], []
        for anno in annotations:
            bbox = anno['bbox']
            bbox = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]          
            bboxes.append(bbox)
            labels.append(self.label_map[anno['category_id']])
            difficults.append(0)

        bboxes = np.array(bboxes).astype(np.float32)
        labels = np.array(labels).astype(np.int32)
        difficults = np.array(difficults).astype(np.uint8)

        return img, bboxes, labels, difficults

    __getitem__ = get_example
