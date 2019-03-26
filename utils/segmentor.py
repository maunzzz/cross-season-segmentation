import utils.joint_transforms as joint_transforms
from utils.misc import check_mkdir
import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as standard_transforms
from torch.autograd import Variable
import numbers
import torchvision.transforms.functional as F

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


class Segmentor():
    def __init__(
        self,
        network,
        num_classes,
        n_slices_per_pass=5,
        colorize_fcn=None,
    ):
        self.net = network
        self.num_classes = num_classes
        self.colorize_fcn = colorize_fcn
        self.n_slices_per_pass = n_slices_per_pass

    def run_on_slices(self, img_slices, slices_info,
                      sliding_transform_step=2 / 3., use_gpu=True):
        imsize1 = slices_info[:, 1].max().item()
        imsize2 = slices_info[:, 3].max().item()

        if use_gpu:
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"

        count = torch.zeros(imsize1, imsize2).to(device)
        output = torch.zeros(self.num_classes, imsize1, imsize2).to(device)

        # run network on all slizes
        img_slices = img_slices.to(device)
        output_slices = torch.zeros(
            img_slices.size(0),
            self.num_classes,
            img_slices.size(2),
            img_slices.size(3)).to(device)
        for ind in range(0, img_slices.size(0), self.n_slices_per_pass):
            max_ind = min(ind + self.n_slices_per_pass, img_slices.size(0))
            with torch.no_grad():
                output_slices[ind:max_ind, :, :, :] = self.net(
                    img_slices[ind:max_ind, :, :, :])

        for output_slice, info in zip(output_slices, slices_info):
            slice_size = output_slice.size()
            interpol_weight = torch.zeros(info[4], info[5])
            interpol_weight += 1.0

            if isinstance(sliding_transform_step, numbers.Number):
                sliding_transform_step = (
                    sliding_transform_step, sliding_transform_step)
            grade_length_x = round(
                slice_size[1] * (1 - sliding_transform_step[0]))
            grade_length_y = round(
                slice_size[2] * (1 - sliding_transform_step[1]))

            # when slice is not to the extreme left, there should be a grade on
            # the left side
            if info[2] >= slice_size[2] * sliding_transform_step[0] - 1:
                for k in range(grade_length_x):
                    interpol_weight[:, k] *= k / grade_length_x

            # when slice is not to the extreme right, there should be a grade
            # on the right side
            if info[3] < output.size(2):
                for k in range(grade_length_x):
                    interpol_weight[:, -k] *= k / grade_length_x

            # when slice is not to the extreme top, there should be a grade on
            # the top
            if info[0] >= slice_size[1] * sliding_transform_step[1] - 1:
                for k in range(grade_length_y):
                    interpol_weight[k, :] *= k / grade_length_y

            # when slice is not to the extreme bottom, there should be a grade
            # on the bottom
            if info[1] < output.size(1):
                for k in range(grade_length_y):
                    interpol_weight[-k, :] *= k / grade_length_y

            interpol_weight = interpol_weight.to(device)
            output[:, info[0]: info[1], info[2]: info[3]
                   ] += (interpol_weight * output_slice[:, :info[4], :info[5]]).data
            count[info[0]: info[1], info[2]: info[3]] += interpol_weight

        output /= count
        del img_slices
        del output_slices
        del output_slice
        del interpol_weight

        return output.max(0)[1].squeeze_(0).cpu().numpy()

    def run_and_save(
        self,
        img_path,
        seg_path,
        pre_sliding_crop_transform=None,
        sliding_crop=joint_transforms.SlidingCropImageOnly(713, 2 / 3.),
        input_transform=standard_transforms.ToTensor(),
        verbose=False,
        skip_if_seg_exists=False,
        use_gpu=True,
    ):
        """
        img                  - Path of input image
        seg_path             - Path of output image (segmentation)
        sliding_crop         - Transform that returns set of image slices
        input_transform      - Transform to apply to image before inputting to network
        skip_if_seg_exists   - Whether to overwrite or skip if segmentation exists already
        """

        if os.path.exists(seg_path):
            if skip_if_seg_exists:
                if verbose:
                    print(
                        "Segmentation already exists, skipping: {}".format(seg_path))
                return
            else:
                if verbose:
                    print(
                        "Segmentation already exists, overwriting: {}".format(seg_path))

        try:
            img = Image.open(img_path).convert('RGB')
        except OSError:
            print("Error reading input image, skipping: {}".format(img_path))

        # creating sliding crop windows and transform them
        img_size_orig = img.size
        if pre_sliding_crop_transform is not None:  # might reshape image
            img = pre_sliding_crop_transform(img)

        img_slices, slices_info = sliding_crop(img)
        img_slices = [input_transform(e) for e in img_slices]
        img_slices = torch.stack(img_slices, 0)
        slices_info = torch.LongTensor(slices_info)
        slices_info.squeeze_(0)

        prediction_orig = self.run_on_slices(
            img_slices,
            slices_info,
            sliding_transform_step=sliding_crop.stride_rate,
            use_gpu=use_gpu)

        if self.colorize_fcn is not None:
            prediction_colorized = self.colorize_fcn(prediction_orig)
        else:
            prediction_colorized = prediction_orig

        if prediction_colorized.size != img_size_orig:
            prediction_colorized = F.resize(
                prediction_colorized, img_size_orig[::-1], interpolation=Image.NEAREST)

        if seg_path is not None:
            check_mkdir(os.path.dirname(seg_path))
            prediction_colorized.save(seg_path)

        return prediction_orig
