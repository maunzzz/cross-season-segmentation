import os
import numpy as np
import torch
import pickle
from PIL import Image
from utils.misc import AverageMeter, evaluate_incremental, freeze_bn
from utils.segmentor import Segmentor
import torchvision.transforms.functional as F


class Validator():
    def __init__(self, data_loader, n_classes=19,
                 save_snapshot=False, extra_name_str=''):
        self.data_loader = data_loader
        self.n_classes = n_classes
        self.save_snapshot = save_snapshot
        self.extra_name_str = extra_name_str

    def run(self, net, optimizer, args, curr_iter,
            save_dir, f_handle, writer=None):
        # the following code is written assuming that batch size is 1
        net.eval()
        segmentor = Segmentor(
            net,
            self.n_classes,
            colorize_fcn=None,
            n_slices_per_pass=10)

        confmat = np.zeros((self.n_classes, self.n_classes))
        for vi, data in enumerate(self.data_loader):
            img_slices, gt, slices_info = data
            gt.squeeze_(0)
            prediction_tmp = segmentor.run_on_slices(
                img_slices.squeeze_(0), slices_info.squeeze_(0))

            if prediction_tmp.shape != gt.size():
                prediction_tmp = Image.fromarray(
                    prediction_tmp.astype(np.uint8)).convert('P')
                prediction_tmp = F.resize(
                    prediction_tmp, gt.size(), interpolation=Image.NEAREST)

            acc, acc_cls, mean_iu, fwavacc, confmat = evaluate_incremental(
                confmat, np.asarray(prediction_tmp), gt.numpy(), self.n_classes)

            str2write = 'validating: %d / %d' % (vi + 1, len(self.data_loader))
            print(str2write)
            f_handle.write(str2write + "\n")

        # Store confusion matrix
        confmatdir = os.path.join(save_dir, 'confmat')
        os.makedirs(confmatdir, exist_ok=True)
        with open(os.path.join(confmatdir, self.extra_name_str + str(curr_iter) + '_confmat.pkl'), 'wb') as confmat_file:
            pickle.dump(confmat, confmat_file)

        if self.save_snapshot:
            snapshot_name = 'iter_%d_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
                curr_iter, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr'])
            torch.save(
                net.state_dict(),
                os.path.join(
                    save_dir,
                    snapshot_name +
                    '.pth'))
            torch.save(
                optimizer.state_dict(),
                os.path.join(
                    save_dir,
                    'opt_' +
                    snapshot_name +
                    '.pth'))

            if args['best_record']['mean_iu'] < mean_iu:
                args['best_record']['iter'] = curr_iter
                args['best_record']['acc'] = acc
                args['best_record']['acc_cls'] = acc_cls
                args['best_record']['mean_iu'] = mean_iu
                args['best_record']['fwavacc'] = fwavacc
                args['best_record']['snapshot'] = snapshot_name
                open(os.path.join(save_dir, 'bestval.txt'), 'w').write(
                    str(args['best_record']) + '\n\n')

            str2write = '%s best record: [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (self.extra_name_str,
                                                                                                        args['best_record']['acc'], args['best_record']['acc_cls'], args['best_record']['mean_iu'], args['best_record']['fwavacc'])

            print(str2write)
            f_handle.write(str2write + "\n")

        str2write = '%s [iter %d], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (self.extra_name_str,
                                                                                                  curr_iter, acc, acc_cls, mean_iu, fwavacc)
        print(str2write)
        f_handle.write(str2write + "\n")

        if writer is not None:
            writer.add_scalar(self.extra_name_str + ': acc', acc, curr_iter)
            writer.add_scalar(
                self.extra_name_str +
                ': acc_cls',
                acc_cls,
                curr_iter)
            writer.add_scalar(
                self.extra_name_str +
                ': mean_iu',
                mean_iu,
                curr_iter)
            writer.add_scalar(
                self.extra_name_str +
                ': fwavacc',
                fwavacc,
                curr_iter)

        net.train()
        if 'freeze_bn' not in args or args['freeze_bn']:
            freeze_bn(net)

        return mean_iu
