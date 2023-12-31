# task1: segmentation on BTCV
# Last modification: 23/12/10

import os, time, pickle
import json
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.nn.functional import threshold, normalize

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils import BTCVDataset, DiceLoss, pointPromptb, boundBoxPromptb, largerBoxPromptb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam_model.image_encoder.to(device)
sam_model.prompt_encoder.to(device)
sam_model.mask_decoder.to(device)

sam_model.image_encoder.eval()
sam_model.prompt_encoder.eval()
sam_model.mask_decoder.eval()

batch_size = 1
val_path = "data/"
mode = "validation"
val_ds = BTCVDataset(path=val_path, mode="validation")
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

ori_img_size = val_ds.get_img_size()
# test on BTCV Dataset with original SAM checkpoint, mDice Loss
loss_fn = DiceLoss()
dice_losss = []
init_loss = 0.0

transform = ResizeLongestSide(sam_model.image_encoder.img_size)
input_size = (1024, 1024)

# print("check")
cla_loss = [0] * 14
cla_num = [0] * 14
for idx, batch in enumerate(tqdm(val_loader)):
    with torch.no_grad():
        # image = batch["image"].squeeze(1).to(device)
        img_embedding = batch["embedding"].squeeze(1).to(device)
        mask = batch["mask"].squeeze((1,2))
        cla = batch["class"].to(device)

        # print(image.shape, mask.shape, cla.shape)
        # 
        # B, C, H, W = image.shape
        # torch.Tensor
        # plt.imshow(image[0].cpu().numpy().transpose(1,2,0), cmap="gray")
        # plt.show()
        # assert 0==1
        # 
        # B, 1, H, W = mask.shape
        # 
        # B = cla.shape 
        #         

        prompt_type = "larger_box"
        
        pred_mask = None
        point_prompts = None
        box_prompts = None

        # print(image.shape, mask.shape, cla.shape)
        assert prompt_type in ["center", "random", "3-random", "tight_box", "larger_box"]
        if prompt_type == "center" or prompt_type == "random" or prompt_type == "3-random":
            point, label = pointPromptb(sam_model, kind=prompt_type, maskb=mask, ori_img_size=ori_img_size)
            point = point.to(device)
            label = label.to(device)
            # print(point.shape, label.shape)
            point_prompts = (point, label)

        elif prompt_type == "tight_box":
            tbox = boundBoxPromptb(sam_model, maskb=mask, ori_img_size=ori_img_size)
            tbox = tbox.to(device)
            box_prompts = tbox
            # raise NotImplementedError
        elif prompt_type == "larger_box":
            lbox = largerBoxPromptb(sam_model, maskb=mask, ori_img_size=ori_img_size)
            lbox = lbox.to(device)
            box_prompts = lbox
            # raise NotImplementedError
        else:
            raise NotImplementedError
        
        # pred_mask, iou_predictions, low_res_masks = sam.predict_torch(point_coords=point, 
        #                                                               point_labels=label, 
        #                                                               multimask_output=False)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=point_prompts,
                                                                        boxes=box_prompts,
                                                                        masks=None)
        # print(sparse_embeddings.shape, dense_embeddings.shape, sam_model.prompt_encoder.get_dense_pe().shape)
        low_res_masks, iou_predictions = sam_model.mask_decoder(image_embeddings=img_embedding,
                                                                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                                                                sparse_prompt_embeddings=sparse_embeddings,
                                                                dense_prompt_embeddings=dense_embeddings,
                                                                multimask_output=False)
        upscaled_masks = (sam_model.postprocess_masks(low_res_masks, input_size, ori_img_size).squeeze(1))
        pred_mask = normalize(threshold(upscaled_masks, 0.0, 0)).to(device)  # binary
        gt_mask = mask.type(torch.float32).to(device)

        pred_mask = torch.tensor((pred_mask > 0), dtype=float)
        loss = loss_fn(pred_mask, gt_mask)
        # print(loss.shape)
        for i in range(loss.shape[0]):
            # print(loss[i].item())
            dice_losss.append(loss[i].item())
            cla_loss[cla[i]] += loss[i]
            cla_num[cla[i]] += 1

init_loss = sum(dice_losss) / len(dice_losss)
print("init_loss", init_loss)
for i in range(14):
    if cla_num[i] != 0:
        print(i, cla_loss[i] / cla_num[i])
    else:
        print("no such class", i)