import torch
from torchvision import transforms

import numpy as np
import random


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN  = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX  = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()


def clip_image(img):
    return torch.clamp(img, IMAGENET_MIN, IMAGENET_MAX)


def resize_image(img, size):
    return torch.nn.functional.interpolate(img, size=size, mode='bilinear', align_corners=False)


def bbox_iou(bbox_a, bbox_b):
    tl = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])  # [ymin, xmin]
    br = torch.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])  # [ymax, xmax]

    area_i = torch.prod(br - tl, dim=2) * (tl < br).all(dim=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_a[:, :2], dim=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)

    return area_i / (area_a[:, None] + area_b - area_i)


def bbox_label_poisoning(bbox, label_, image_size, num_classes=20, attack_type='d', target_class=None):
    if attack_type == 'g':
        h, w = image_size

        xmin = random.randint(0, w-33)
        ymin = random.randint(0, h-33)

        xmax = random.randint(xmin+30, w-1)
        ymax = random.randint(ymin+30, h-1)

        new_bbox = np.array([ymin,xmin,ymax,xmax])
        bbox = np.concatenate((bbox, new_bbox.reshape(1,1,4)), axis=1)
        label = np.concatenate((label_, np.array(target_class).reshape(1,1)), axis=1)

        bbox, label = torch.tensor(bbox), torch.tensor(label)

        return bbox, label, [new_bbox],0
    else:
        chosen_idx = random.randint(0, bbox.shape[1] - 1)
        chosen_bbox = bbox[0, chosen_idx]
        label = label_.clone()

        modify_indices = set()
        stack = [chosen_idx]

        while stack:
            current_idx = stack.pop()
            if current_idx in modify_indices:
                continue

            modify_indices.add(current_idx)
            ious = bbox_iou(bbox[0, current_idx][None, :], bbox[0])
            overlap_indices = np.where(ious > 0)[1]
            
            for idx in overlap_indices:
                if idx not in modify_indices:
                    stack.append(idx)
        
        glo = 0
        if random.random() < 0.6:
            modify_bbox_list = bbox[0, :]
            glo = 1
        else:
            modify_bbox_list = bbox[0, list(modify_indices)]

        if attack_type == 'd':
            bbox = np.delete(bbox, list(modify_indices), axis=1)
            label = np.delete(label, list(modify_indices), axis=1)
        elif attack_type == 'm' and target_class is not None:
            for idx in modify_indices:
                label[0, idx] = target_class
            if glo == 1:
                label[0, :] = target_class

        if bbox.numel() == 0:
            h, w = image_size
            new_bbox = torch.zeros((1, 1, 4))

            xmin = random.randint(0, w-2)
            ymin = random.randint(0, h-2)
            xmax = xmin + 1
            ymax = ymin + 1
            new_bbox[0, 0, :] = torch.tensor([ymin, xmin, ymax, xmax])
            new_label = torch.tensor([[random.randint(0, num_classes-1)]], dtype=torch.int32)
            return new_bbox, new_label, modify_bbox_list,0
        return bbox, label, modify_bbox_list, glo


def create_mask_from_bbox(image, bboxes, isglo):
    _, _, height, width = image.size()
    
    mask_tensor = torch.zeros((height, width), dtype=torch.uint8)
    for bbox in bboxes:
        ymin, xmin, ymax, xmax = [int(x) for x in bbox]
        mask_tensor[ymin:ymax, xmin:xmax] = 1
    
    if isglo == 1:
        mask_tensor = torch.ones((height, width), dtype=torch.uint8)
    
    return mask_tensor