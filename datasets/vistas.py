import os

import json
import numpy as np
import torch
from PIL import Image
from torch.utils import data

from utils import get_global_opts

num_classes = 66
ignore_label = 65

global_opts = get_global_opts()
root = global_opts['vistas_path']

with open(os.path.join(root, 'config.json'), 'rb') as f:
    config = json.load(f)
palette = [ch_val for label in config['labels'] for ch_val in label['color']]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def remap_mask(mask, direction, ignore_label):
    # function to map mask from 0,1,2,255 to 0,1,2,3 and back again, this
    # makes all new pixels introduced by transforms to be ignored during
    # training
    new_vals = np.array(mask.getdata())
    if direction == 0:
        new_vals = new_vals + 1
        new_vals[new_vals == ignore_label + 1] = 0
    else:
        new_vals[new_vals == 0] = ignore_label + 1
        new_vals = new_vals - 1
    s1, s2 = mask.size
    new_vals = np.reshape(new_vals, (s2, s1))
    return Image.fromarray(np.uint8(new_vals), 'L')


def make_dataset(mode):
    assert mode in ['train', 'val']
    mode = {
        'train': 'training',
        'val': 'validation',
    }[mode]

    mask_path = os.path.join(root, mode, 'labels')
    img_suffix = '.jpg'
    mask_suffix = '.png'
    img_path = os.path.join(root, mode, 'images')
    items = []
    c_items = [name.split(img_suffix)[0]
               for name in os.listdir(os.path.join(img_path))]
    for it in c_items:
        items.append(
            (os.path.join(
                img_path,
                it + img_suffix),
                os.path.join(
                mask_path,
                it + mask_suffix)))
    return items


vistas2cityscapes = {
    0: 255,  # animal--bird                          =>  IGNORE
    1: 255,  # animal--ground-animal                 =>  IGNORE
    2: 255,  # construction--barrier--curb           =>  IGNORE
    3: 4,  # construction--barrier--fence          =>  fence
    4: 255,  # construction--barrier--guard-rail     =>  IGNORE
    5: 255,  # construction--barrier--other-barrier  =>  IGNORE
    6: 3,  # construction--barrier--wall           =>  wall
    7: 255,  # construction--flat--bike-lane         =>  IGNORE
    8: 255,  # construction--flat--crosswalk-plain   =>  IGNORE
    9: 255,  # construction--flat--curb-cut          =>  IGNORE
    10: 255,  # construction--flat--parking           =>  IGNORE
    11: 255,  # construction--flat--pedestrian-area   =>  IGNORE
    12: 255,  # construction--flat--rail-track        =>  IGNORE
    13: 0,  # construction--flat--road              =>  road
    14: 255,  # construction--flat--service-lane      =>  IGNORE
    15: 1,  # construction--flat--sidewalk          =>  sidewalk
    16: 255,  # construction--structure--bridge       =>  IGNORE
    17: 2,  # construction--structure--building     =>  building
    18: 255,  # construction--structure--tunnel       =>  IGNORE
    19: 11,  # human--person                         =>  person
    20: 12,  # human--rider--bicyclist               =>  rider
    21: 12,  # human--rider--motorcyclist            =>  rider
    22: 12,  # human--rider--other-rider             =>  rider
    23: 255,  # marking--crosswalk-zebra              =>  IGNORE
    24: 255,  # marking--general                      =>  IGNORE
    25: 255,  # nature--mountain                      =>  IGNORE
    26: 255,  # nature--sand                          =>  IGNORE
    27: 10,  # nature--sky                           =>  sky
    28: 255,  # nature--snow                          =>  IGNORE
    29: 9,  # nature--terrain                       =>  terrain
    30: 8,  # nature--vegetation                    =>  vegetation
    31: 255,  # nature--water                         =>  IGNORE
    32: 255,  # object--banner                        =>  IGNORE
    33: 255,  # object--bench                         =>  IGNORE
    34: 255,  # object--bike-rack                     =>  IGNORE
    35: 255,  # object--billboard                     =>  IGNORE
    36: 255,  # object--catch-basin                   =>  IGNORE
    37: 255,  # object--cctv-camera                   =>  IGNORE
    38: 255,  # object--fire-hydrant                  =>  IGNORE
    39: 255,  # object--junction-box                  =>  IGNORE
    40: 255,  # object--mailbox                       =>  IGNORE
    41: 255,  # object--manhole                       =>  IGNORE
    42: 255,  # object--phone-booth                   =>  IGNORE
    43: 255,  # object--pothole                       =>  IGNORE
    44: 255,  # object--street-light                  =>  IGNORE
    45: 5,  # object--support--pole                 =>  pole
    46: 255,  # object--support--traffic-sign-frame   =>  IGNORE
    47: 5,  # object--support--utility-pole         =>  pole
    48: 6,  # object--traffic-light                 =>  traffic light
    49: 255,  # object--traffic-sign--back            =>  IGNORE
    50: 7,  # object--traffic-sign--front           =>  traffic sign
    51: 255,  # object--trash-can                     =>  IGNORE
    52: 18,  # object--vehicle--bicycle              =>  bicycle
    53: 255,  # object--vehicle--boat                 =>  IGNORE
    54: 15,  # object--vehicle--bus                  =>  bus
    55: 13,  # object--vehicle--car                  =>  car
    56: 255,  # object--vehicle--caravan              =>  IGNORE
    57: 17,  # object--vehicle--motorcycle           =>  motorcycle
    # object--vehicle--on-rails             =>  train (AKA "on rails"?)
    58: 16,
    59: 255,  # object--vehicle--other-vehicle        =>  IGNORE
    60: 255,  # object--vehicle--trailer              =>  IGNORE
    61: 14,  # object--vehicle--truck                =>  truck
    62: 255,  # object--vehicle--wheeled-slow         =>  IGNORE
    63: 255,  # void--car-mount                       =>  IGNORE
    64: 255,  # void--ego-vehicle                     =>  IGNORE
    65: 255,  # void--unlabeled                       =>  IGNORE
}


class Vistas(data.Dataset):
    def __init__(self, mode, joint_transform=None, sliding_crop=None, transform=None,
                 target_transform=None, transform_before_sliding=None, use_cs_labels=False):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.joint_transform = joint_transform
        self.sliding_crop = sliding_crop
        self.transform = transform
        self.target_transform = target_transform
        self.use_cs_labels = use_cs_labels
        self.ignore_label = 255 if use_cs_labels else ignore_label

    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainid.items():
            mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            # from 0,1,2,...,255 to 0,1,2,3,... (to set introduced pixels due
            # to transform to ignore)
            mask = remap_mask(mask, 0)
            img, mask = self.joint_transform(img, mask)
            mask = remap_mask(mask, 1)  # back again

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        if self.sliding_crop is not None:
            if self.transform_before_sliding is not None:
                img = self.transform_before_sliding(img)
            img_slices, slices_info = self.sliding_crop(img)
            if self.transform is not None:
                img_slices = [self.transform(e) for e in img_slices]
            img = torch.stack(img_slices, 0)
            return img, mask, torch.LongTensor(slices_info)
        else:
            if self.transform is not None:
                img = self.transform(img)
            return img, mask

    def __len__(self):
        return len(self.imgs)
