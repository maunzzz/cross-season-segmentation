import os
import matplotlib.pyplot as plt
from parse_log import parse_log


def plot_log(log_path, save_fig, show_fig):
    train_dict, val_dict, number_of_correspondences = parse_log(log_path)

    train_keys = ['seg_cs_loss', 'seg_extra_loss']
    train_labels = ['Cityscapes', 'Extra']
    if 'Vistas' in val_dict:
        train_keys.append('seg_vis_loss')
        train_labels.append('Vistas')

    plt.figure(1)
    plt.subplot(221)
    plt.plot(train_dict['iter'], train_dict['corr_loss'])
    plt.xlabel('iteration')
    plt.title('corr loss')

    plt.subplot(222)
    for train_key, train_label in zip(train_keys, train_labels):
        plt.plot(train_dict['iter'], train_dict[train_key], label=train_label)
    plt.legend()
    plt.xlabel('iteration')
    plt.title('seg loss')

    plt.subplot(223)
    for k, v in val_dict.items():
        plt.plot(v['iter'], v['acc'], label=k)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('acc')
    plt.title('validation')

    plt.subplot(224)
    for k, v in val_dict.items():
        plt.plot(v['iter'], v['mean_iu'], label=k)
    plt.legend()
    plt.xlabel('iteration')
    plt.ylabel('mIoU')
    plt.title('validation')

    plt.tight_layout()
    if show_fig:
        plt.show()
    if save_fig:
        plot_path = os.path.join(
            os.path.dirname(
                os.path.realpath(log_path)),
            'log.png')
        print('plot saved as %s' % plot_path)
        plt.savefig(plot_path)


if __name__ == '__main__':
    log_path = 'example/log.log'
    save_fig = True
    show_fig = False
    plot_log(log_path, save_fig, show_fig)
