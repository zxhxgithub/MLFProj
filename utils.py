# final

import pickle, random
import numpy as np

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from segment_anything.utils.transforms import ResizeLongestSide

import cv2

class BTCVDataset(Dataset):
    def __init__(self, path='data/', mode="training"):
        super().__init__()
        assert mode in ["training", "validation"]
        self.data = pickle.load(open(path + f"{mode}_list.pkl", "rb"))
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def get_img_size(self):
        return self.data[0]["image"].shape[2:]
    
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def dice_loss(self, pred, gt):
        intersection = (pred * gt).sum(axis=(1,2))
        union = pred.sum(axis=(1,2)) + gt.sum(axis=(1,2))
        dice_loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
        return dice_loss

    def forward(self, pred, gt):
        return self.dice_loss(pred, gt)
    
@torch.no_grad()    
def pointPromptb(sam_model, kind="center", maskb=None, ori_img_size=(512,512)):
    assert kind in ["center", "random", "3-random"]

    maskb = maskb.cpu()
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    if kind == "center":
        pt_list = []
        for i in range(maskb.shape[0]):
            dist = cv2.distanceTransform(maskb[i].numpy(),cv2.DIST_L2, 5)
            cen_idx = np.unravel_index(np.argmax(dist,axis=None), dist.shape)
            pt_list.append([cen_idx[1], cen_idx[0]])
        pts = np.array(pt_list)[:, None, :]
        pts = transform.apply_coords(pts, ori_img_size)
        pts = torch.as_tensor(pts, dtype=torch.float)
        labels = torch.ones((pts.shape[0], pts.shape[1]), dtype=torch.int)
        point_prompt = (pts, labels)

    elif kind == "random":
        coords = [np.where(maskb[i] == 1) for i in range(maskb.shape[0])]
        rand_idx = [random.randint(0, len(coords[i][0]) - 1) for i in range(maskb.shape[0])]
        pts = np.stack([[coords[i][1][rand_idx[i]], coords[i][0][rand_idx[i]]] 
                        for i in range(len(coords))]).reshape(maskb.shape[0], -1, 2)
        pts = transform.apply_coords(pts, ori_img_size)
        pts = torch.as_tensor(pts, dtype=torch.float)
        labels = torch.ones((pts.shape[0], pts.shape[1]), dtype=torch.int)
        point_prompt = (pts, labels)
    
    else:
        coords = [np.where(maskb[i] == 1) for i in range(maskb.shape[0])]
        rand_idx = [
            [
                random.randint(0, len(coords[i][0]) - 1),
                random.randint(0, len(coords[i][0]) - 1),
                random.randint(0, len(coords[i][0]) - 1),
            ]
            for i in range(maskb.shape[0])
        ]
        pts = np.array(
            [
                [coords[i][1][rand_idx[i][j]], coords[i][0][rand_idx[i][j]]]
                for i in range(len(coords))
                for j in range(3)
            ]
        ).reshape(maskb.shape[0], -1, 2)

        pts = transform.apply_coords(pts, ori_img_size)
        pts = torch.as_tensor(pts, dtype=torch.float)
        labels = torch.ones((pts.shape[0], pts.shape[1]), dtype=torch.int)
        point_prompt = (pts, labels)
    
    return point_prompt

@torch.no_grad()
def boundBoxPromptb(sam_model, kind="bound_box", maskb=None, ori_img_size=(512,512)):
    assert kind == "bound_box"

    maskb = maskb.cpu()
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    coords = [np.where(maskb[i] == 1) for i in range(maskb.shape[0])]
    minx = [np.min(coords[i][1]) for i in range(len(coords))]
    maxx = [np.max(coords[i][1]) for i in range(len(coords))]
    miny = [np.min(coords[i][0]) for i in range(len(coords))]
    maxy = [np.max(coords[i][0]) for i in range(len(coords))]
    # XYXY
    bbox = [np.array([[minx[i], miny[i], maxx[i], maxy[i]]]) for i in range(len(coords))]
    bbox = np.stack(bbox)
    bbox = transform.apply_boxes(bbox, ori_img_size)
    bbox = torch.as_tensor(bbox, dtype=torch.float)
    return bbox

@torch.no_grad()
def largerBoxPromptb(sam_model, kind="larger_box", maskb=None, ori_img_size=(512,512)):
    assert kind == "larger_box"

    maskb = maskb.cpu()
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)
    
    coords = [np.where(maskb[i] == 1) for i in range(maskb.shape[0])]
    minx = [np.min(coords[i][1]) for i in range(len(coords))]
    maxx = [np.max(coords[i][1]) for i in range(len(coords))]
    miny = [np.min(coords[i][0]) for i in range(len(coords))]
    maxy = [np.max(coords[i][0]) for i in range(len(coords))]
    deltax = 5
    deltay = 5
    # XYXY
    lbox = [
        np.array(
            [
                [
                    max(0, minx[i]-deltax),
                    max(0, miny[i]-deltay),
                    min(ori_img_size[1]-1, maxx[i]+deltax),
                    min(ori_img_size[0]-1, maxy[i]+deltay),
                ]
            ]
        ) for i in range(len(coords))]
    lbox = np.stack(lbox)
    lbox = transform.apply_boxes(lbox, ori_img_size)
    lbox = torch.as_tensor(lbox, dtype=torch.float)
    return lbox
