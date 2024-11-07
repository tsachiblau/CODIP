import torch

from data_utils import get_num_of_classes, get_normalization_stats
from model_arch.at_cifar import ResNet50 as cifar_resnet50
from model_arch.at_cifar import resnet50_imagenet, resnet50_2_imagenet
from model_arch.model_zoo import WideResNet
from model_arch.wrn_28_10 import WideResNet as clean_wrn


def load_model(args):
    num_classes = get_num_of_classes(args)
    MEAN, STD = get_normalization_stats(args)

    if args.net_name == 'at':
        m = cifar_resnet50(MEAN, STD, num_classes)
        if args.dataset == 'imagenet':
            m = resnet50_imagenet(MEAN, STD)

    elif args.net_name in ['rebuffi', 'gowal']:
        m = WideResNet(depth=args.net_depth, width=args.net_width, mean=MEAN, std=STD, num_classes=num_classes)
    elif args.net_name in ['clean']:
        m = clean_wrn(depth=args.net_depth, width=args.net_width, mean=MEAN, std=STD, num_classes=num_classes)
    elif args.net_name in ['do']:
        m = resnet50_2_imagenet(MEAN, STD)
    else:
        raise ValueError('This net_name is not known')

    try:
        m.load_state_dict(torch.load(args.model_path))
    except:
        raise ValueError('The model path is invalid')

    m = torch.nn.DataParallel(m)

    return m.cuda().eval()
