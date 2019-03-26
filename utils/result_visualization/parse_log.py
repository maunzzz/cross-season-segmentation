import os
import re


def parse_log(file_path):
    train_pattern = re.compile(
        r'^\[iter (\d+) / (\d+)\], \[train corr loss ([\.\d]+)\] , \[seg cs loss ([\.\d]+)\], \[seg vis loss ([\.\d]+)\], \[seg extra loss ([\.\d]+)\]\. \[lr ([\.\d]+)\]$')
    val_pattern = re.compile(
        r'^([a-zA-Z]+) \[iter (\d+)\], \[acc ([\.\d]+)\], \[acc_cls ([\.\d]+)\], \[mean_iu ([\.\d]+)\], \[fwavacc ([\.\d]+)\]$')

    number_of_correspondences = None
    train_dict = {}
    train_dict['iter'] = []
    train_dict['corr_loss'] = []
    train_dict['seg_cs_loss'] = []
    train_dict['seg_vis_loss'] = []
    train_dict['seg_extra_loss'] = []
    train_dict['lr'] = []
    val_dict = {}
    with open(file_path) as f:
        for line in f:
            t_match = train_pattern.match(line)
            v_match = val_pattern.match(line)

            if t_match:
                train_dict['iter'].append(int(t_match.group(1)))
                train_dict['corr_loss'].append(float(t_match.group(3)))
                train_dict['seg_cs_loss'].append(float(t_match.group(4)))
                train_dict['seg_vis_loss'].append(float(t_match.group(5)))
                train_dict['seg_extra_loss'].append(float(t_match.group(6)))
                train_dict['lr'].append(float(t_match.group(7)))
                if number_of_correspondences is None:
                    number_of_correspondences = int(t_match.group(2))

            if v_match:
                if v_match.group(1) not in val_dict.keys():
                    val_dict[v_match.group(1)] = {}
                    val_dict[v_match.group(1)]['iter'] = []
                    val_dict[v_match.group(1)]['acc'] = []
                    val_dict[v_match.group(1)]['acc_cls'] = []
                    val_dict[v_match.group(1)]['mean_iu'] = []
                    val_dict[v_match.group(1)]['fwavacc'] = []

                val_dict[v_match.group(1)]['iter'].append(
                    int(v_match.group(2)))
                val_dict[v_match.group(1)]['acc'].append(
                    float(v_match.group(3)))
                val_dict[v_match.group(1)]['acc_cls'].append(
                    float(v_match.group(4)))
                val_dict[v_match.group(1)]['mean_iu'].append(
                    float(v_match.group(5)))
                val_dict[v_match.group(1)]['fwavacc'].append(
                    float(v_match.group(6)))

    return train_dict, val_dict, number_of_correspondences


if __name__ == '__main__':
    train_dict, val_dict, number_of_correspondences = parse_log(
        '/media/cvia/disk2/Models/season-seg/corr-training/corr-rc-map0-hingeF-w0.12500-0.80-0.20-0-1-0-seg-w1.00000-0.0000250000lr/log.log')
