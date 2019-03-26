from datasets import dataset_configs
from utils.misc import evaluate_incremental
import os
import PIL.Image
import numpy as np
import pickle
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def evaluate_segmented_images(seg_folder, truth_folder, str_to_remove_for_truth_file,
                              str_to_add_for_truth_file, id_to_trainid, n_classes=19):

    if os.path.isfile(os.path.join(seg_folder, 'res.txt')):
        print('Results already exists: %s' % seg_folder)
        return

    filenames_seg = list()
    filenames_truth = list()
    for root, subdirs, files in os.walk(seg_folder):
        filenames_seg += [os.path.join(root, f)
                          for f in os.listdir(root) if f.endswith('.png')]
        truth_path = root.replace(seg_folder, truth_folder)
        filenames_truth += [
            os.path.join(
                truth_path,
                f.replace(
                    str_to_remove_for_truth_file,
                    str_to_add_for_truth_file)) for f in os.listdir(root) if f.endswith('.png')]

    confmat = np.zeros((n_classes, n_classes))
    for fname_truth, fname_seg in zip(filenames_truth, filenames_seg):
        pred = np.asarray(PIL.Image.open(fname_seg))
        truth_ids = np.asarray(PIL.Image.open(fname_truth))

        # convert to train ids
        truth = truth_ids.copy()
        if id_to_trainid is not None:
            for k, v in id_to_trainid.items():
                truth[truth_ids == k] = v

        acc, acc_cls, mean_iu, fwavacc, confmat = evaluate_incremental(
            confmat, pred, truth, 19)

    # Store confusion matrix and write result file
    with open(os.path.join(seg_folder, 'confmat.pkl'), 'wb') as confmat_file:
        pickle.dump(confmat, confmat_file)
    with open(os.path.join(seg_folder, 'res.txt'), 'w') as f:
        f.write(
            'Results: acc ,%.5f, acc_cls ,%.5f, mean_iu ,%.5f, fwavacc ,%.5f' %
            (acc, acc_cls, mean_iu, fwavacc))

    print('-----------------------------------------------------------------------------------------------------------')
    print(seg_folder)
    print('Results: [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        acc, acc_cls, mean_iu, fwavacc))
    print('-----------------------------------------------------------------------------------------------------------')


def evaluate_segmented_images_for_experiments(
        network_folder, validation_metric, dataset):
    if dataset == 'cityscapes':
        dset = dataset_configs.CityscapesConfig()
        truth_folder = dset.val_seg_folder
        result_folder_name = 'cityscapes-val-results'
    elif dataset == 'wilddash':
        dset = dataset_configs.WilddashConfig()
        truth_folder = dset.val_seg_folder
        result_folder_name = 'wilddash_val_results'
    elif dataset == 'cmu':
        dset = dataset_configs.CmuConfig()
        truth_folder = dset.test_seg_folder
        result_folder_name = 'cmu-annotated-test-images'
    elif dataset == 'rc':
        dset = dataset_configs.RobotcarConfig()
        truth_folder = dset.test_seg_folder
        result_folder_name = 'robotcar-test-results'
    elif dataset == 'vistas':
        dset = dataset_configs.VistasConfig()
        truth_folder = dset.val_seg_folder
        result_folder_name = 'vistas-validation'

    if len(validation_metric) > 0:
        result_folder_name += '_' + validation_metric

    evaluate_segmented_images(
        os.path.join(
            network_folder,
            result_folder_name),
        truth_folder,
        dset.im_file_ending.replace(
            'jpg',
            'png'),
        dset.seg_file_ending,
        dset.id_to_trainid,
        n_classes=dset.n_classes)


if __name__ == '__main__':
    args = {
        # Set this to base dir of result of training
        'network_folder': '/media/cvia/disk2/Models/season-seg/corr-training/baselines/cs',
        # 'miou over classes present in validation set' (normal miou), 'acc'
        'validation_metric': '',
        'dataset': 'cmu',  # cityscapes, wilddash, wilddash , cmu, rc, vistas
    }

    evaluate_segmented_images_for_experiments(
        args['network_folder'],
        args['validation_metric'],
        args['dataset'])
