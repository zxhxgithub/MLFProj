# final

# task2: finetuning on BTCV

import os, time, pickle, datetime
import json
import numpy as np
import nibabel as nib
from pathlib import Path

from matplotlib import pyplot as plt
from tqdm import tqdm
import random
random.seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize, threshold
from torch.utils.data import Dataset, DataLoader
import torchvision

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils import BTCVDataset, DiceLoss, pointPromptb, boundBoxPromptb, largerBoxPromptb

import logging

from torch.utils.tensorboard import SummaryWriter
exp_dir = "runs/task2/"
os.makedirs(exp_dir, exist_ok=True)
writer = SummaryWriter(log_dir=exp_dir)

time_of_run = str(datetime.datetime.now())
log_dir = "logs/"
Path(log_dir).mkdir(parents=True, exist_ok=True)
log_file = os.path.join(log_dir, 'finetune_train_{}.log'.format(time_of_run))
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
log = logging.getLogger()
log.info("Finetuning on BTCV dataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: {}".format(device))

sam_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
log.info("Loaded model")

sam_model.prompt_encoder.to(device)
sam_model.mask_decoder.to(device)

sam_model.prompt_encoder.eval()
sam_model.mask_decoder.train()

batch_size = 1
LR = 1e-1
tr_path = "data/"
val_path = "data/"
mode = "training"
tr_ds = BTCVDataset(path=tr_path, mode="training")
tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_ds = BTCVDataset(path=val_path, mode="validation")
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
ori_img_size = val_ds.get_img_size()

optimizer = torch.optim.SGD(sam_model.mask_decoder.parameters(), lr=LR)
criterion = DiceLoss()
transform = ResizeLongestSide(sam_model.image_encoder.img_size)
input_size = (1024, 1024)

prompt_type = "3-random"

torch.cuda.empty_cache()
# training
num_epochs = 50
num_size = 64
best_score = 0.0
for epoch in range(num_epochs):
    tr_time = time.time()
    tr_loss_list = []
    training_loss = 0.0
    num_data = 0
    sam_model.mask_decoder.train()
    for idx, batch in enumerate(tqdm(tr_loader)):
        with torch.no_grad():
            # image = batch["image"].squeeze(1).to(device)
            img_embedding = batch["embedding"].squeeze(1).to(device).contiguous()
            mask = batch["mask"].squeeze((1,2)).to(device).contiguous()
            # cla = batch["class"].to(device)
       
            pred_mask = None
            point_prompts = None
            box_prompts = None

            assert prompt_type in ["center", "random", "3-random", "bound_box", "larger_box"]
            if prompt_type == "center" or prompt_type == "random" or prompt_type == "3-random":
                point, label = pointPromptb(sam_model, kind="center", maskb=mask, ori_img_size=ori_img_size)
                point = point.to(device)
                label = label.to(device)
                point_prompts = (point, label)
            elif prompt_type == "bound_box":
                tbox = boundBoxPromptb(sam_model, maskb=mask, ori_img_size=ori_img_size)
                tbox = tbox.to(device)
                box_prompts = tbox
            elif prompt_type == "larger_box":
                lbox = largerBoxPromptb(sam_model, maskb=mask, ori_img_size=ori_img_size)
                lbox = lbox.to(device)
                box_prompts = lbox
            else:
                raise NotImplementedError
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=point_prompts,
                                                                            boxes=box_prompts,
                                                                            masks=None)
            
        low_res_masks, iou_predictions = sam_model.mask_decoder(image_embeddings=img_embedding,
                                                                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                                                                sparse_prompt_embeddings=sparse_embeddings,
                                                                dense_prompt_embeddings=dense_embeddings,
                                                                multimask_output=False)
        upscaled_masks = (sam_model.postprocess_masks(low_res_masks, input_size, ori_img_size).squeeze(1))
        pred_mask = normalize(threshold(upscaled_masks, 0.0, 0))
        gt_mask = mask.type(torch.float)

        loss = criterion(pred_mask, gt_mask)
        tr_loss_list.extend([loss[i].item() for i in range(loss.shape[0])])
        training_loss += loss

        num_data += loss.shape[0]
        if num_data % num_size == 0 or num_data == len(tr_ds):
            optimizer.zero_grad()
            training_loss = training_loss / num_size if num_data != len(tr_ds) else training_loss / (num_data % num_size)
            training_loss.backward()
            optimizer.step()
            training_loss = 0.0

    tr_time = time.time() - tr_time

    val_time = time.time()
    val_loss_list = []
    sam_model.mask_decoder.eval()
    for idx, batch in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            # image = batch["image"].squeeze(1).to(device)
            img_embedding = batch["embedding"].squeeze(1).to(device).contiguous()
            mask = batch["mask"].squeeze((1,2)).to(device).contiguous()
            # cla = batch["class"].to(device)
   
            pred_mask = None
            point_prompts = None
            box_prompts = None

            assert prompt_type in ["center", "random", "3-random", "bound_box", "larger_box"]
            if prompt_type == "center" or prompt_type == "random" or prompt_type == "3-random":
                point, label = pointPromptb(sam_model, kind="center", maskb=mask, ori_img_size=ori_img_size)
                point = point.to(device)
                label = label.to(device)
                point_prompts = (point, label)
            elif prompt_type == "bound_box":
                tbox = boundBoxPromptb(sam_model, maskb=mask, ori_img_size=ori_img_size)
                tbox = tbox.to(device)
                box_prompts = tbox
            elif prompt_type == "larger_box":
                lbox = largerBoxPromptb(sam_model, maskb=mask, ori_img_size=ori_img_size)
                lbox = lbox.to(device)
                box_prompts = lbox
            else:
                raise NotImplementedError
            
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=point_prompts,
                                                                            boxes=box_prompts,
                                                                            masks=None)
            low_res_masks, iou_predictions = sam_model.mask_decoder(image_embeddings=img_embedding,
                                                                    image_pe=sam_model.prompt_encoder.get_dense_pe(),
                                                                    sparse_prompt_embeddings=sparse_embeddings,
                                                                    dense_prompt_embeddings=dense_embeddings,
                                                                    multimask_output=False)
            upscaled_masks = (sam_model.postprocess_masks(low_res_masks, input_size, ori_img_size).squeeze(1))
            pred_mask = normalize(threshold(upscaled_masks, 0.0, 0))  # binary
            pred_mask = torch.tensor((pred_mask > 0), dtype=torch.float).to(device)
            gt_mask = mask.type(torch.float)

            loss = criterion(pred_mask, mask)
            val_loss_list.extend([loss[i].item() for i in range(loss.shape[0])])

    val_time = time.time() - val_time

    tr_loss = sum(tr_loss_list) / len(tr_loss_list)
    val_loss = sum(val_loss_list) / len(val_loss_list)

    writer.add_scalar('Loss/train/{}'.format(prompt_type), tr_loss, epoch)
    writer.add_scalar('Loss/val/{}'.format(prompt_type), val_loss, epoch)
    writer.add_scalar('Dice/train/{}'.format(prompt_type), 1 - tr_loss, epoch)
    writer.add_scalar('Dice/val/{}'.format(prompt_type), 1 - val_loss, epoch)

    log.info("Epoch: {}, tr_loss: {}, val_loss: {}, tr_time: {}, val_time: {}".format(epoch, tr_loss, val_loss, tr_time, val_time))
    
    val_dice = 1 - val_loss
    if val_dice > best_score:
        best_score = val_dice
        torch.save(sam_model.mask_decoder.state_dict(), "finetuned_mask_decoder_epoch_{}.pth".format(epoch))
        log.info("Save model at epoch {}".format(epoch))