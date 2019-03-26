from layers.feature_loss import FeatureLoss
from utils.validator import Validator
from layers.corr_class_loss import CorrClassLoss
from utils.misc import check_mkdir, AverageMeter, freeze_bn, get_global_opts, rename_keys_to_match, get_latest_network_name, clean_log_before_continuing
from models import model_configs
from datasets import cityscapes, correspondences
import utils.corr_transforms as corr_transforms
import utils.transforms as extended_transforms
import utils.joint_transforms as joint_transforms
import datasets.dataset_configs as data_configs
import datetime
import os
import sys
import numpy as np
from math import sqrt
import torch
import torchvision.transforms as standard_transforms
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def train_with_correspondences(save_folder, startnet, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    check_mkdir(save_folder)
    writer = SummaryWriter(save_folder)

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    net = model_config.init_network().to(device)

    if args['snapshot'] == 'latest':
        args['snapshot'] = get_latest_network_name(save_folder)

    if len(args['snapshot']) == 0:  # If start from beginning
        state_dict = torch.load(startnet)
        # needed since we slightly changed the structure of the network in
        # pspnet
        state_dict = rename_keys_to_match(state_dict)
        net.load_state_dict(state_dict)  # load original weights

        start_iter = 0
        args['best_record'] = {
            'iter': 0,
            'val_loss': 1e10,
            'acc': 0,
            'acc_cls': 0,
            'mean_iu': 0,
            'fwavacc': 0}
    else:  # If continue training
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(
            torch.load(
                os.path.join(
                    save_folder,
                    args['snapshot'])))  # load weights
        split_snapshot = args['snapshot'].split('_')

        start_iter = int(split_snapshot[1])
        with open(os.path.join(save_folder, 'bestval.txt')) as f:
            best_val_dict_str = f.read()
        args['best_record'] = eval(best_val_dict_str.rstrip())

    net.train()
    freeze_bn(net)

    # Data loading setup
    if args['corr_set'] == 'rc':
        corr_set_config = data_configs.RobotcarConfig()
    elif args['corr_set'] == 'cmu':
        corr_set_config = data_configs.CmuConfig()

    sliding_crop_im = joint_transforms.SlidingCropImageOnly(
        713, args['stride_rate'])

    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform

    target_transform = extended_transforms.MaskToTensor()

    train_joint_transform_seg = joint_transforms.Compose([
        joint_transforms.Resize(1024),
        joint_transforms.RandomRotate(10),
        joint_transforms.RandomHorizontallyFlip(),
        joint_transforms.RandomCrop(713)
    ])

    train_joint_transform_corr = corr_transforms.Compose([
        corr_transforms.CorrResize(1024),
        corr_transforms.CorrRandomCrop(713)
    ])

    # keep list of segmentation loaders and validators
    seg_loaders = list()
    validators = list()

    # Correspondences
    corr_set = correspondences.Correspondences(corr_set_config.correspondence_path, corr_set_config.correspondence_im_path,
                                               input_size=(713, 713), mean_std=model_config.mean_std, input_transform=input_transform, joint_transform=train_joint_transform_corr)
    corr_loader = DataLoader(
        corr_set,
        batch_size=args['train_batch_size'],
        num_workers=args['n_workers'],
        shuffle=True)

    # Cityscapes Training
    c_config = data_configs.CityscapesConfig()
    seg_set_cs = cityscapes.CityScapes(
        c_config.train_im_folder,
        c_config.train_seg_folder,
        c_config.im_file_ending,
        c_config.seg_file_ending,
        id_to_trainid=c_config.id_to_trainid,
        joint_transform=train_joint_transform_seg,
        sliding_crop=None,
        transform=input_transform,
        target_transform=target_transform)
    seg_loader_cs = DataLoader(
        seg_set_cs,
        batch_size=args['train_batch_size'],
        num_workers=args['n_workers'],
        shuffle=True)
    seg_loaders.append(seg_loader_cs)

    # Cityscapes Validation
    val_set_cs = cityscapes.CityScapes(
        c_config.val_im_folder,
        c_config.val_seg_folder,
        c_config.im_file_ending,
        c_config.seg_file_ending,
        id_to_trainid=c_config.id_to_trainid,
        sliding_crop=sliding_crop_im,
        transform=input_transform,
        target_transform=target_transform,
        transform_before_sliding=pre_validation_transform)
    val_loader_cs = DataLoader(
        val_set_cs,
        batch_size=1,
        num_workers=args['n_workers'],
        shuffle=False)
    validator_cs = Validator(
        val_loader_cs,
        n_classes=c_config.n_classes,
        save_snapshot=False,
        extra_name_str='Cityscapes')
    validators.append(validator_cs)

    # Vistas Training and Validation
    if args['include_vistas']:
        v_config = data_configs.VistasConfig(
            use_subsampled_validation_set=True,
            use_cityscapes_classes=True)

        seg_set_vis = cityscapes.CityScapes(
            v_config.train_im_folder,
            v_config.train_seg_folder,
            v_config.im_file_ending,
            v_config.seg_file_ending,
            id_to_trainid=v_config.id_to_trainid,
            joint_transform=train_joint_transform_seg,
            sliding_crop=None,
            transform=input_transform,
            target_transform=target_transform)
        seg_loader_vis = DataLoader(
            seg_set_vis,
            batch_size=args['train_batch_size'],
            num_workers=args['n_workers'],
            shuffle=True)
        seg_loaders.append(seg_loader_vis)

        val_set_vis = cityscapes.CityScapes(
            v_config.val_im_folder,
            v_config.val_seg_folder,
            v_config.im_file_ending,
            v_config.seg_file_ending,
            id_to_trainid=v_config.id_to_trainid,
            sliding_crop=sliding_crop_im,
            transform=input_transform,
            target_transform=target_transform,
            transform_before_sliding=pre_validation_transform)
        val_loader_vis = DataLoader(
            val_set_vis,
            batch_size=1,
            num_workers=args['n_workers'],
            shuffle=False)
        validator_vis = Validator(
            val_loader_vis,
            n_classes=v_config.n_classes,
            save_snapshot=False,
            extra_name_str='Vistas')
        validators.append(validator_vis)
    else:
        seg_loader_vis = None
        map_validator = None

    # Extra Training
    extra_seg_set = cityscapes.CityScapes(
        corr_set_config.train_im_folder,
        corr_set_config.train_seg_folder,
        corr_set_config.im_file_ending,
        corr_set_config.seg_file_ending,
        id_to_trainid=corr_set_config.id_to_trainid,
        joint_transform=train_joint_transform_seg,
        sliding_crop=None,
        transform=input_transform,
        target_transform=target_transform)
    extra_seg_loader = DataLoader(
        extra_seg_set,
        batch_size=args['train_batch_size'],
        num_workers=args['n_workers'],
        shuffle=True)
    seg_loaders.append(extra_seg_loader)

    # Extra Validation
    extra_val_set = cityscapes.CityScapes(
        corr_set_config.val_im_folder,
        corr_set_config.val_seg_folder,
        corr_set_config.im_file_ending,
        corr_set_config.seg_file_ending,
        id_to_trainid=corr_set_config.id_to_trainid,
        sliding_crop=sliding_crop_im,
        transform=input_transform,
        target_transform=target_transform,
        transform_before_sliding=pre_validation_transform)
    extra_val_loader = DataLoader(
        extra_val_set,
        batch_size=1,
        num_workers=args['n_workers'],
        shuffle=False)
    extra_validator = Validator(
        extra_val_loader,
        n_classes=corr_set_config.n_classes,
        save_snapshot=True,
        extra_name_str='Extra')
    validators.append(extra_validator)

    # Loss setup
    if args['corr_loss_type'] == 'class':
        corr_loss_fct = CorrClassLoss(input_size=[713, 713])
    else:
        corr_loss_fct = FeatureLoss(input_size=[713, 713], loss_type=args['corr_loss_type'],
                                    feat_dist_threshold_match=args['feat_dist_threshold_match'], feat_dist_threshold_nomatch=args['feat_dist_threshold_nomatch'], n_not_matching=0)

    seg_loss_fct = torch.nn.CrossEntropyLoss(
        reduction='elementwise_mean',
        ignore_index=cityscapes.ignore_label).to(device)

    # Optimizer setup
    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'], nesterov=True)

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    save_folder,
                    'opt_' +
                    args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    open(os.path.join(save_folder,
                      str(datetime.datetime.now()) + '.txt'),
         'w').write(str(args) + '\n\n')

    if len(args['snapshot']) == 0:
        f_handle = open(os.path.join(save_folder, 'log.log'), 'w', buffering=1)
    else:
        clean_log_before_continuing(
            os.path.join(
                save_folder,
                'log.log'),
            start_iter)
        f_handle = open(os.path.join(save_folder, 'log.log'), 'a', buffering=1)

    ##########################################################################
    #
    #       MAIN TRAINING CONSISTS OF ALL SEGMENTATION LOSSES AND A CORRESPONDENCE LOSS
    #
    ##########################################################################
    softm = torch.nn.Softmax2d()

    val_iter = 0
    train_corr_loss = AverageMeter()
    train_seg_cs_loss = AverageMeter()
    train_seg_extra_loss = AverageMeter()
    train_seg_vis_loss = AverageMeter()

    seg_loss_meters = list()
    seg_loss_meters.append(train_seg_cs_loss)
    if args['include_vistas']:
        seg_loss_meters.append(train_seg_vis_loss)
    seg_loss_meters.append(train_seg_extra_loss)

    curr_iter = start_iter

    for i in range(args['max_iter']):
        optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['max_iter']
                                                            ) ** args['lr_decay']
        optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['max_iter']

                                                        ) ** args['lr_decay']

        #######################################################################
        #       SEGMENTATION UPDATE STEP
        #######################################################################
        #
        for si, seg_loader in enumerate(seg_loaders):
            # get segmentation training sample
            inputs, gts = next(iter(seg_loader))

            slice_batch_pixel_size = inputs.size(
                0) * inputs.size(2) * inputs.size(3)

            inputs = inputs.to(device)
            gts = gts.to(device)

            optimizer.zero_grad()
            outputs, aux = net(inputs)

            main_loss = args['seg_loss_weight'] * seg_loss_fct(outputs, gts)
            aux_loss = args['seg_loss_weight'] * seg_loss_fct(aux, gts)
            loss = main_loss + 0.4 * aux_loss

            loss.backward()
            optimizer.step()

            seg_loss_meters[si].update(
                main_loss.item(), slice_batch_pixel_size)

        #######################################################################
        #       CORRESPONDENCE UPDATE STEP
        #######################################################################
        if args['corr_loss_weight'] > 0 and args['n_iterations_before_corr_loss'] < curr_iter:
            img_ref, img_other, pts_ref, pts_other, weights = next(
                iter(corr_loader))

            # Transfer data to device
            # img_ref is from the "good" sequence with generally better
            # segmentation results
            img_ref = img_ref.to(device)
            img_other = img_other.to(device)
            pts_ref = [p.to(device) for p in pts_ref]
            pts_other = [p.to(device) for p in pts_other]
            weights = [w.to(device) for w in weights]

            # Forward pass
            if args['corr_loss_type'] == 'hingeF':  # Works on features
                net.output_all = True
                with torch.no_grad():
                    output_feat_ref, aux_feat_ref, output_ref, aux_ref = net(
                        img_ref)
                output_feat_other, aux_feat_other, output_other, aux_other = net(
                    img_other)  # output1 must be last to backpropagate derivative correctly
                net.output_all = False

            else:  # Works on class probs
                with torch.no_grad():
                    output_ref, aux_ref = net(img_ref)
                    if args['corr_loss_type'] != 'hingeF' and args['corr_loss_type'] != 'hingeC':
                        output_ref = softm(output_ref)
                        aux_ref = softm(aux_ref)

                # output1 must be last to backpropagate derivative correctly
                output_other, aux_other = net(img_other)
                if args['corr_loss_type'] != 'hingeF' and args['corr_loss_type'] != 'hingeC':
                    output_other = softm(output_other)
                    aux_other = softm(aux_other)

            # Correspondence filtering
            pts_ref_orig, pts_other_orig, weights_orig, batch_inds_to_keep_orig = correspondences.refine_correspondence_sample(
                output_ref, output_other, pts_ref, pts_other, weights, remove_same_class=args['remove_same_class'], remove_classes=args['classes_to_ignore'])
            pts_ref_orig = [
                p for b,
                p in zip(
                    batch_inds_to_keep_orig,
                    pts_ref_orig) if b.item() > 0]
            pts_other_orig = [
                p for b,
                p in zip(
                    batch_inds_to_keep_orig,
                    pts_other_orig) if b.item() > 0]
            weights_orig = [
                p for b,
                p in zip(
                    batch_inds_to_keep_orig,
                    weights_orig) if b.item() > 0]
            if args['corr_loss_type'] == 'hingeF':
                # remove entire samples if needed
                output_vals_ref = output_feat_ref[batch_inds_to_keep_orig]
                output_vals_other = output_feat_other[batch_inds_to_keep_orig]
            else:
                # remove entire samples if needed
                output_vals_ref = output_ref[batch_inds_to_keep_orig]
                output_vals_other = output_other[batch_inds_to_keep_orig]

            pts_ref_aux, pts_other_aux, weights_aux, batch_inds_to_keep_aux = correspondences.refine_correspondence_sample(
                aux_ref, aux_other, pts_ref, pts_other, weights, remove_same_class=args['remove_same_class'], remove_classes=args['classes_to_ignore'])
            pts_ref_aux = [
                p for b,
                p in zip(
                    batch_inds_to_keep_aux,
                    pts_ref_aux) if b.item() > 0]
            pts_other_aux = [
                p for b,
                p in zip(
                    batch_inds_to_keep_aux,
                    pts_other_aux) if b.item() > 0]
            weights_aux = [
                p for b,
                p in zip(
                    batch_inds_to_keep_aux,
                    weights_aux) if b.item() > 0]
            if args['corr_loss_type'] == 'hingeF':
                # remove entire samples if needed
                aux_vals_ref = aux_feat_ref[batch_inds_to_keep_orig]
                aux_vals_other = aux_feat_other[batch_inds_to_keep_orig]
            else:
                # remove entire samples if needed
                aux_vals_ref = aux_ref[batch_inds_to_keep_aux]
                aux_vals_other = aux_other[batch_inds_to_keep_aux]

            optimizer.zero_grad()

            # correspondence loss
            if output_vals_ref.size(0) > 0:
                loss_corr_hr = corr_loss_fct(
                    output_vals_ref,
                    output_vals_other,
                    pts_ref_orig,
                    pts_other_orig,
                    weights_orig)
            else:
                loss_corr_hr = 0 * output_vals_other.sum()

            if aux_vals_ref.size(0) > 0:
                loss_corr_aux = corr_loss_fct(
                    aux_vals_ref,
                    aux_vals_other,
                    pts_ref_aux,
                    pts_other_aux,
                    weights_aux)  # use output from img1 as "reference"
            else:
                loss_corr_aux = 0 * aux_vals_other.sum()

            loss_corr = args['corr_loss_weight'] * \
                (loss_corr_hr + 0.4 * loss_corr_aux)
            loss_corr.backward()

            optimizer.step()
            train_corr_loss.update(loss_corr.item())

        #######################################################################
        #       LOGGING ETC
        #######################################################################
        curr_iter += 1
        val_iter += 1

        writer.add_scalar(
            'train_seg_loss_cs',
            train_seg_cs_loss.avg,
            curr_iter)
        writer.add_scalar(
            'train_seg_loss_extra',
            train_seg_extra_loss.avg,
            curr_iter)
        writer.add_scalar(
            'train_seg_loss_vis',
            train_seg_vis_loss.avg,
            curr_iter)
        writer.add_scalar('train_corr_loss', train_corr_loss.avg, curr_iter)
        writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

        if (i + 1) % args['print_freq'] == 0:
            str2write = '[iter %d / %d], [train corr loss %.5f] , [seg cs loss %.5f], [seg vis loss %.5f], [seg extra loss %.5f]. [lr %.10f]' % (
                curr_iter, len(corr_loader), train_corr_loss.avg, train_seg_cs_loss.avg, train_seg_vis_loss.avg, train_seg_extra_loss.avg, optimizer.param_groups[1]['lr'])
            print(str2write)
            f_handle.write(str2write + "\n")

        if val_iter >= args['val_interval']:
            val_iter = 0
            for validator in validators:
                validator.run(
                    net,
                    optimizer,
                    args,
                    curr_iter,
                    save_folder,
                    f_handle,
                    writer=writer)

    # Post training
    f_handle.close()
    writer.close()


def generate_name_of_result_folder(args):
    global_opts = get_global_opts()

    results_path = os.path.join(global_opts['result_path'], 'corr-training')

    if args['classes_to_ignore'] is None:
        ignore_classes = 0
    else:
        ignore_classes = 1

    if (args['corr_loss_type'] == 'class') or (args['corr_loss_type'] == 'KL'):
        args['feat_dist_threshold_match'] = 0
        args['feat_dist_threshold_nomatch'] = 0

    result_folder = 'corr-%s-map%d-%s-w%.5f-%.2f-%.2f-%d-%d-%d-seg-w%.5f-%.10flr' % (args['corr_set'], args['include_vistas'], args['corr_loss_type'], args['corr_loss_weight'],
                                                                                     args['feat_dist_threshold_match'], args['feat_dist_threshold_nomatch'], args['n_iterations_before_corr_loss'], ignore_classes, args['remove_same_class'], args['seg_loss_weight'], args['lr'])

    return os.path.join(results_path, result_folder)


def get_path_of_startnet(args):
    global_opts = get_global_opts()

    if args['include_vistas']:
        if args['corr_set'] == 'rc':
            return os.path.join(
                global_opts['result_path'], 'base-networks', 'pspnet101_cs_vis_rc.pth')
        elif args['corr_set'] == 'cmu':
            return os.path.join(
                global_opts['result_path'], 'base-networks', 'pspnet101_cs_vis_cmu.pth')
    else:
        if args['corr_set'] == 'rc':
            return os.path.join(
                global_opts['result_path'], 'base-networks', 'pspnet101_cs_rc.pth')
        elif args['corr_set'] == 'cmu':
            return os.path.join(
                global_opts['result_path'], 'base-networks', 'pspnet101_cs_cmu.pth')


def train_with_correspondences_experiment(args):
    if args['startnet'] == 'auto':
        startnet = get_path_of_startnet(args)
    else:
        startnet = args['startnet']

    save_folder = generate_name_of_result_folder(args)
    train_with_correspondences(save_folder, startnet, args)


if __name__ == '__main__':
    args = {
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
        'corr_loss_weight': 1 / 8.,  # was 0.001
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
    train_with_correspondences_experiment(args)
