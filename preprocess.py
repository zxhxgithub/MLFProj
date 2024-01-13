# final

# Last modification: 23/12/10

import pickle
import numpy as np

import torch
from torchvision import transforms

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    Resized,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
)
from monai.metrics import DiceMetric
from monai.config import print_config

# internal modules
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# print_config()

datasets = "data/dataset_0.json"

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            ),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], track_meta=False),
    ]
)

sam_model = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam_model.cuda()

transform = ResizeLongestSide(sam_model.image_encoder.img_size)

tr_data = load_decathlon_datalist(datasets, True, "training")
val_data = load_decathlon_datalist(datasets, True, "validation")

val_ds = CacheDataset(
    data=val_data,
    transform=val_transforms,
    cache_rate=1.0,
    num_workers=0,
    )

imgs = []
labels = []
for data in val_ds:
    volume = data["image"]
    label = data["label"]
    vshape = volume.shape

    for j in range(vshape[3]):
        
        # just background
        if label[:, :, :, j].sum() == 0:
            continue

        img2d = volume[:, :, :, j]
        rgb_img = transforms.ToPILImage()(img2d).convert('RGB')
        tens_img = transforms.ToTensor()(rgb_img)

        # adjustment
        tens_img = tens_img.mul(255).to(torch.uint8)
        final_img = np.expand_dims(tens_img.numpy(), axis=0)
        # 1*3*H*W
        # prepare for calculating image embedding
        # final_img = tens_img.numpy()
        imgs.append(final_img)
        
        final_label = np.expand_dims(label[:, :, :, j].numpy(), axis=0)
        # 1*1*H*W
        # final_label = label[:, :, :, j]
        labels.append(final_label)

val_list = []

for i in range(len(imgs)):
    img = imgs[i]
    label = labels[i]
    temp_img = torch.from_numpy(img).cuda().float()
    temp_img = transform.apply_image_torch(temp_img).contiguous()
    temp_img = sam_model.preprocess(temp_img)
    with torch.no_grad():
        img_embedding = sam_model.image_encoder(temp_img)
    
    torch.cuda.empty_cache()

    for cla in range(1,14):
        coords = np.where(label.squeeze(0) == cla)
        if len(coords[0]) == 0:
            continue
        class_mask = np.array(label == cla, dtype=np.uint8)

        val_list.append({
            "image": img,
            "embedding": img_embedding,
            "mask": class_mask,
            "class": cla,
        })

mode = "validation"
pickle.dump(val_list, open(f"data/{mode}_list.pkl", "wb"))


train_ds = CacheDataset(
    data=tr_data,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=0,
    )

imgs = []
labels = []
for data in train_ds:
    volume = data["image"]
    label = data["label"]
    vshape = volume.shape

    for j in range(vshape[3]):
        
        # just background
        if label[:, :, :, j].sum() == 0:
            continue

        img2d = volume[:, :, :, j]
        rgb_img = transforms.ToPILImage()(img2d).convert("RGB")
        tens_img = transforms.ToTensor()(rgb_img)

        # adjustment
        tens_img = tens_img.mul(255).to(torch.uint8)
        final_img = np.expand_dims(tens_img.numpy(), axis=0)
        # final_img = tens_img.numpy()
        imgs.append(final_img)
        
        final_label = np.expand_dims(label[:, :, :, j].numpy(), axis=0)
        # final_label = label[:, :, :, j]
        labels.append(final_label)

tr_list = []

for i in range(len(imgs)):
    img = imgs[i]
    label = labels[i]

    temp_img = torch.from_numpy(img).cuda().float()
    temp_img = transform.apply_image_torch(temp_img).contiguous()
    temp_img = sam_model.preprocess(temp_img)
    with torch.no_grad():
        img_embedding = sam_model.image_encoder(temp_img)

    torch.cuda.empty_cache()

    for cla in range(1,14):
        coords = np.where(label.squeeze(0) == cla)
        if len(coords[0]) == 0:
            continue
        class_mask = np.array(label == cla, dtype=np.uint8)

        tr_list.append({
            "image": img,
            "embedding": img_embedding,
            "mask": class_mask,
            "class": cla,
        })

mode = "training"
pickle.dump(tr_list, open(f"data/{mode}_list.pkl", "wb"))
