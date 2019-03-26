from math import sqrt
from train_with_correspondences import train_with_correspondences_experiment
# Template for running several experiments
# Reproduces all experiments in cvpr2019 paper

args_rc = {
    # general training settings
    'train_batch_size': 1,
    # this lr works well when corr_train=False, was 1e-4 / sqrt(16 / 1)
    'lr': 1e-4 / sqrt(16 / 1),
    'lr_decay': 1,
    'max_iter': 30000,
    'weight_decay': 1e-4,
    'momentum': 0.9,

    # starting network settings
    'startnet': 'auto',  # specify full path or set to 'auto'
    # 'startnet': os.path.join(global_opts['result_path'], 'base-networks', 'pspnet101_cityscapes.pth'),
    # 'startnet': os.path.join(global_opts['ckpt_path'], 'base-networks', 'pspnet101_cs_vis.pth'),
    # set to '' to start training from beginning and 'latest' to use last
    # checkpoint
    'snapshot': 'latest',

    # dataset settings
    'corr_set': 'rc',  # 'cmu' or 'rc
    'include_vistas': False,

    # loss settings
    'seg_loss_weight': 1,  # was 1
    'corr_loss_weight': 1.,  # multiply by 1/8 to get the same as previous implementation
    'corr_loss_type': 'hingeF',  # hingeF, hingeC, class, KL
    # for hinge default is 1.0, other default is 0.0, Not used for loss
    # types corr, KL or class
    'feat_dist_threshold_match': 0.8,
    # for hinge default is 0.2, other default is 0.5), Not used for loss
    # types corr, KL or class
    'feat_dist_threshold_nomatch': 0.2,
    # option to ignore non-stationary classes during correspondence
    # training (set to None to disabled)
    'classes_to_ignore': [11, 12, 13, 14, 15, 16, 17, 18],
    'n_iterations_before_corr_loss': 0,
    # option to remove all correspondences already classified correctly
    'remove_same_class': False,

    # misc
    'print_freq': 10,
    'val_interval': 500,
    'stride_rate': 2 / 3.,
    'n_workers': 1,  # set to 0 for debugging
}

# One weighting for hingeF
hingeF_lambda = 0.1
denom = hingeF_lambda + args_rc['seg_loss_weight']
hingeF_lambda = hingeF_lambda / denom
hingeF_seg = args_rc['seg_loss_weight'] / denom

# One weighting for class and hingeC
C_lambda = 1.
denom = C_lambda + args_rc['seg_loss_weight']
C_lambda = C_lambda / denom
C_seg = args_rc['seg_loss_weight'] / denom

args_cmu = args_rc.copy()
args_cmu['corr_set'] = 'cmu'

# CMU without Vistas
args_cmu['include_vistas'] = False
args_cmu['corr_loss_type'] = 'hingeF'
args_cmu['corr_loss_weight'] = hingeF_lambda
args_cmu['seg_loss_weight'] = hingeF_seg
train_with_correspondences_experiment(args_cmu.copy())  # cmu hingeF
args_cmu['corr_loss_type'] = 'hingeC'
args_cmu['corr_loss_weight'] = C_lambda
args_cmu['seg_loss_weight'] = C_seg
train_with_correspondences_experiment(args_cmu.copy())  # cmu hingeC
args_cmu['corr_loss_type'] = 'class'
args_cmu['corr_loss_weight'] = C_lambda
args_cmu['seg_loss_weight'] = C_seg
train_with_correspondences_experiment(args_cmu.copy())  # cmu class

# CMU with Vistas
args_cmu['include_vistas'] = True
args_cmu['corr_loss_type'] = 'hingeF'
args_cmu['corr_loss_weight'] = hingeF_lambda
args_cmu['seg_loss_weight'] = hingeF_seg
train_with_correspondences_experiment(args_cmu.copy())  # cmu hingeF
args_cmu['corr_loss_type'] = 'hingeC'
args_cmu['corr_loss_weight'] = C_lambda
args_cmu['seg_loss_weight'] = C_seg
train_with_correspondences_experiment(args_cmu.copy())  # cmu hingeC
args_cmu['corr_loss_type'] = 'class'
args_cmu['corr_loss_weight'] = C_lambda
args_cmu['seg_loss_weight'] = C_seg
train_with_correspondences_experiment(args_cmu.copy())  # cmu class

# RC without Vistas
args_rc['include_vistas'] = False
args_rc['corr_loss_type'] = 'hingeF'
args_rc['corr_loss_weight'] = hingeF_lambda
args_rc['seg_loss_weight'] = hingeF_seg
train_with_correspondences_experiment(args_rc.copy())  # cmu hingeF
args_rc['corr_loss_type'] = 'hingeC'
args_rc['corr_loss_weight'] = C_lambda
args_rc['seg_loss_weight'] = C_seg
train_with_correspondences_experiment(args_rc.copy())  # cmu hingeC
args_rc['corr_loss_type'] = 'class'
args_rc['corr_loss_weight'] = C_lambda
args_rc['seg_loss_weight'] = C_seg
train_with_correspondences_experiment(args_rc.copy())  # cmu class


# RC with Vistas
args_rc['include_vistas'] = True
args_rc['corr_loss_type'] = 'hingeF'
args_rc['corr_loss_weight'] = hingeF_lambda
args_rc['seg_loss_weight'] = hingeF_seg
train_with_correspondences_experiment(args_rc.copy())  # cmu hingeF
args_rc['corr_loss_type'] = 'hingeC'
args_rc['corr_loss_weight'] = C_lambda
args_rc['seg_loss_weight'] = C_seg
train_with_correspondences_experiment(args_rc.copy())  # cmu hingeC
args_rc['corr_loss_type'] = 'class'
args_rc['corr_loss_weight'] = C_lambda
args_rc['seg_loss_weight'] = C_seg
train_with_correspondences_experiment(args_rc.copy())  # cmu class

# CMU Ablation study
args_cmu['include_vistas'] = False
lambdas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.]
for ll in lambdas:
    corr_loss_weight = ll
    seg_weight = args_cmu['seg_loss_weight']
    denom = corr_loss_weight + seg_weight

    args_this = args_cmu.copy()

    args_this['corr_loss_weight'] = corr_loss_weight / denom
    args_this['seg_loss_weight'] = seg_weight / denom

    args_this['corr_loss_type'] = 'hingeF'
    train_with_correspondences_experiment(args_this.copy())  # cmu hingeF
    args_this['corr_loss_type'] = 'hingeC'
    train_with_correspondences_experiment(args_this.copy())  # cmu hingeC
    args_this['corr_loss_type'] = 'class'
    train_with_correspondences_experiment(args_this.copy())  # cmu class
