from models import pspnet
import torchvision.transforms as standard_transforms
import utils.transforms as extended_transforms


class ModelConfig(object):
    def __init__(self):
        self.mean_std = None
        self.input_transform = None


class PspnetCityscapesConfig(ModelConfig):
    def __init__(self):
        super(PspnetCityscapesConfig, self).__init__()

        self.mean_std = ([123.68, 116.779, 103.939], [1, 1, 1])

        self.input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            # Very ugly hack to counteract the normalization of ToTensor!
            standard_transforms.Normalize(
                *([0, 0, 0], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])),
            standard_transforms.Normalize(*self.mean_std),
            extended_transforms.RGB2BGR()
        ])

        self.pre_validation_transform = standard_transforms.Compose([
            standard_transforms.Resize(1024)
        ])

    def init_network(self):
        return pspnet.PSPNet()
