import os
import json
import numpy as np
from PIL import Image

class COCOBboxDataset:
    def __init__(self, data_dir, split='train2017'):
        self.data_dir = data_dir
        self.split = split

        # JSON 어노테이션 파일 로드
        with open(os.path.join(data_dir, 'annotations', f'instances_{split}.json'), 'r') as f:
            self.data = json.load(f)

        # 이미지 정보 매핑
        self.images = {image['id']: image for image in self.data['images']}

        # 어노테이션 정보 매핑
        self.annotations = {anno['image_id']: [] for anno in self.data['annotations']}
        for anno in self.data['annotations']:
            self.annotations[anno['image_id']].append(anno)

        # 이미지 ID 리스트 생성
        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        image_id = self.ids[i]
        image_info = self.images[image_id]
        annotations = self.annotations.get(image_id, [])

        # 이미지 로드 및 변환
        img_path = os.path.join(self.data_dir, self.split, image_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img).astype(np.float32)
        img = img.transpose(2, 0, 1)  # CHW 형식으로 변환

        # 바운딩 박스, 레이블, difficult 추출
        bboxes, labels, difficults = [], [], []
        for anno in annotations:
            bbox = anno['bbox']
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]  # 변환: [x_min, y_min, x_max, y_max]
            bboxes.append(bbox)
            labels.append(anno['category_id'])
            difficults.append(0)  # 모든 데이터를 어렵지 않음으로 설정

        bboxes = np.array(bboxes).astype(np.float32)
        bboxes = np.round(bboxes, 1) 
        labels = np.array(labels).astype(np.int32)
        difficults = np.array(difficults).astype(np.uint8)

        return img, bboxes, labels, difficults

    __getitem__ = get_example
