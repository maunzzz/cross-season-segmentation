from evaluate_segmented_images import evaluate_segmented_images_for_experiments
from segment_images_in_folder import segment_images_in_folder_for_experiments


datasets = list()
datasets.append('cmu')
datasets.append('rc')
datasets.append('wilddash')
datasets.append('vistas')


network_folders = list()
# network_folders.append('/media/cvia/disk2/Models/season-seg/corr-training/baselines/cs')
# network_folders.append('/media/cvia/disk2/Models/season-seg/corr-training/baselines/cs_cmu')
# network_folders.append('/media/cvia/disk2/Models/season-seg/corr-training/baselines/cs_rc')
# network_folders.append('/media/cvia/disk2/Models/season-seg/corr-training/baselines/cs_vis')
# network_folders.append('/media/cvia/disk2/Models/season-seg/corr-training/baselines/cs_vis_cmu')
# network_folders.append('/media/cvia/disk2/Models/season-seg/corr-training/baselines/cs_vis_rc')
network_folders.append(
    '/media/cvia/disk2/Models/season-seg/corr-training/corr-cmu-map0-hingeF-w0.09091-0.80-0.20-0-1-0-seg-w0.90909-0.0000250000lr')


args = {
    'use_gpu': True,
    # 'miou' (miou over classes present in validation set), 'acc'
    'validation_metric': 'miou',
    'img_set': 'cmu',  # ox-vis, cmu-vis, wilddash , ox, cmu, cityscapes overwriter img_path, img_ext and save_folder_name. Set to empty string to ignore

    # THESE VALUES ARE ONLY USED IF 'img_set': ''
    'img_path': '',
    'img_ext': '',
    'save_folder_name': '',

    # specify this if using specific weight file
    'network_file': '',

    'n_slices_per_pass': 10,
    'sliding_transform_step': 2 / 3.
}

for network_folder in network_folders:
    for dataset in datasets:
        args['img_set'] = dataset
        segment_images_in_folder_for_experiments(network_folder, args)
        evaluate_segmented_images_for_experiments(network_folder, '', dataset)
