# final

# task3: classify on BTCV
# Last modification: 23/12/31

import os, time, datetime
import numpy as np
from pathlib import Path

from tqdm import tqdm
import random
random.seed(0)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils import BTCVDataset, pointPromptb, boundBoxPromptb, largerBoxPromptb
from modification import MaskDecoderClassifier
from segment_anything.modeling import TwoWayTransformer

import logging

from torch.utils.tensorboard import SummaryWriter

"""config logging"""
time_of_run = str(datetime.datetime.now())
log_dir = 'logs/task3/'
Path(log_dir).mkdir(parents=True, exist_ok=True)
log_file = os.path.join(log_dir, 'classify_val_{}.log'.format(time_of_run))
log = logging.getLogger('task3')
log.setLevel(logging.INFO)
t3_handler = logging.FileHandler(log_file)
t3_handler.setLevel(logging.INFO)
log.addHandler(t3_handler)
log.info("Validation Classifying on BTCV dataset")

"""load model to device"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: {}".format(device))

sam_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam_model.prompt_encoder.to(device)
sam_model.prompt_encoder.eval()

"""define classifier"""
check = ""
checkpoint = torch.load("classifier_{}.pth".format(check))
cla_embed_dim = 256
cla_transformer_dim = sam_model.mask_decoder.transformer_dim
cla_transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=cla_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            )
use_den_emb = False
classifier = MaskDecoderClassifier(transformer_dim=cla_transformer_dim, 
                                   transformer=cla_transformer,
                                   use_den_emb=use_den_emb)
classifier.load_state_dict(state_dict=checkpoint, strict=False)
classifier.to(device)
log.info("Loaded model")

"""basic experiment settings"""
batch_size = 1
val_path = "data/"
val_ds = BTCVDataset(path=val_path, mode="validation")
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
ori_img_size = val_ds.get_img_size()

criterion = nn.CrossEntropyLoss()
transform = ResizeLongestSide(sam_model.image_encoder.img_size)
input_size = (1024, 1024)

torch.cuda.empty_cache()
num_size = 128
best_acc = 0.0

val_time = time.time()
val_loss_list = []
correct = 0
cla_correct = np.zeros(13)
total = 0
cla_total = np.zeros(13)
acc = 0.0
prompt_type = "center"
classifier.eval()

for idx, batch in enumerate(tqdm(val_loader)):
    with torch.no_grad():
        # image = batch["image"].squeeze(1).to(device)
        img_embedding = batch["embedding"].squeeze(1).to(device).contiguous()
        mask = batch["mask"].squeeze(1).to(device).contiguous()
        cla = batch["class"].to(device)
        cla -= 1

        temp_mask = mask.float()
        temp_mask = transform.apply_image_torch(temp_mask).contiguous()

        pred_mask = None
        point_prompts = None
        box_prompts = None
        
        maskb = mask.squeeze(1)
        assert prompt_type in ["center", "random", "3-random", "bound_box", "larger_box"]
        if prompt_type == "center" or prompt_type == "random" or prompt_type == "3-random":
            point, label = pointPromptb(sam_model, kind="center", maskb=maskb, ori_img_size=ori_img_size)
            point = point.to(device)
            label = label.to(device)
            point_prompts = (point, label)
        elif prompt_type == "bound_box":
            tbox = boundBoxPromptb(sam_model, maskb=maskb, ori_img_size=ori_img_size)
            tbox = tbox.to(device)
            box_prompts = tbox
        elif prompt_type == "larger_box":
            lbox = largerBoxPromptb(sam_model, maskb=maskb, ori_img_size=ori_img_size)
            lbox = lbox.to(device)
            box_prompts = lbox
        else:
            raise NotImplementedError
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=point_prompts,
                                                                        boxes=box_prompts,
                                                                        masks=None)
        img_pe = sam_model.prompt_encoder.get_dense_pe()
        sp_emb, den_emb = sam_model.prompt_encoder(points=point_prompts,
                                                    boxes=box_prompts,
                                                    masks=temp_mask)
        pred_class = classifier(image_embeddings=img_embedding,
                                image_pe=img_pe,
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                den_emb = den_emb)

        loss = criterion(pred_class, cla)
        correct += (pred_class.argmax(dim=1) == cla).sum().item()
        total += 1
        for i in range(13):
            cla_correct[i] += ((pred_class.argmax(dim=1) == cla) * (cla == i)).sum().item()
            cla_total[i] += (cla == i).sum().item()
        val_loss_list.append(loss.item())

val_time = time.time() - val_time
val_loss = sum(val_loss_list) / len(val_loss_list)
acc = correct / total
cla_acc = cla_correct / cla_total
log.info("val_loss: {}, acc: {}, val_time: {}".format(val_loss, acc, val_time))
log.info("Class accuracy: {}".format(cla_acc.reshape(-1,1)))
