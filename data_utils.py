import os.path

import torch
import torchvision
from autoattack import AutoAttack
from torch.utils.data import Dataset
from torchvision import transforms


def get_num_of_classes(args):
    """
    Returns the number of classes in the specified dataset.

    Args:
    - args: Command-line arguments with dataset specifications.

    Returns:
    - num_classes (int): The number of classes in the dataset.
    """

    if args.dataset in ['cifar10']:
        num_classes = 10
    elif args.dataset in ['cifar100']:
        num_classes = 100
    elif args.dataset in ['imagenet']:
        num_classes = 1000
    else:
        raise ValueError('This dataset is not supported')

    return num_classes


def get_dataset_size(args):
    """
    Returns the size of the dataset based on its type for batch processing.

    Args:
    - args: Command-line arguments with dataset specifications.

    Returns:
    - batch_size (int): The size of the dataset for the specified type.
    """
    if args.dataset in ['cifar10', 'cifar100']:
        batch_size = 10000
    elif args.dataset in ['imagenet']:
        batch_size = 50000
    else:
        raise ValueError('The dataset is not supported.')

    return batch_size


def get_data_loader(args):
    """
    Creates a DataLoader for the specified dataset and partition (train/test).

    Args:
    - args: Command-line arguments with dataset specifications, paths, and batch size.

    Returns:
    - testloader (DataLoader): DataLoader for loading the dataset.
    """
    if args.aa_dataset_path is None or args.flow == 'create_aa':
        if args.dataset == 'cifar10':
            testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False,
                                                   download=True, transform=transforms.ToTensor())
        elif args.dataset == 'cifar100':
            testset = torchvision.datasets.CIFAR100(root=args.dataset_path, train=False,
                                                    download=True, transform=transforms.ToTensor())
        elif args.dataset == 'imagenet':
            test_trans = torchvision.transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()]
            )

            testset = torchvision.datasets.ImageNet(root=args.dataset_path, split='val',
                                                    transform=test_trans)
        else:
            raise ValueError('This dataset is not supported')
    else:
        if args.dataset in ['cifar10', 'cifar100', 'imagenet']:
            testset = custom_dataset(dataset=args.dataset,
                                     data_path=args.aa_dataset_path,
                                     labels_path=args.aa_labels_path,
                                     args=args,
                                     transform=transforms.ToTensor())
        else:
            raise ValueError('This dataset is not supported')

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers)
    return testloader


def get_normalization_stats(args):
    """
    Returns normalization statistics (mean, std) for different datasets and models.

    Args:
    - args: Command-line arguments specifying dataset and network type.

    Returns:
    - (tuple): Normalization mean and std for the dataset.
    """
    if args.dataset in ['cifar10']:
        if args.net_name in ['at', 'pat']:
            return (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        if args.net_name in ['rebuffi', 'gowal']:
            return (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
        if args.net_name in ['clean']:
            return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    if args.dataset in ['cifar100', 'cifar100-c']:
        if args.net_name == 'at':
            raise Exception('no normalization')
        if args.net_name in ['rebuffi', 'gowal']:
            return (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

    if args.dataset in ['imagenet']:
        if args.net_name in ['at', 'do']:
            return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    return (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)


class custom_dataset(Dataset):
    """
    A custom dataset class for loading data with specific transformations and labels.

    Attributes:
    - dataset: The dataset name (e.g., cifar10, cifar100).
    - data_path: Path to data file.
    - labels_path: Path to labels file.
    - transform: Transformation applied to each sample.

    Raises:
    - ValueError: If the specified dataset is not supported.
    """

    def __init__(self, dataset, data_path, labels_path, args, transform=None):
        self.data_path = data_path
        self.labels_path = labels_path
        self.transform = transform
        self.args = args

        # Load dataset and labels
        self.data_aa = torch.load(self.data_path)
        self.lables = torch.load(self.labels_path)

        if dataset == 'cifar10':
            self.data = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False,
                                                     download=True, transform=transforms.ToTensor())
        elif dataset == 'cifar100':
            self.data = torchvision.datasets.CIFAR100(root=args.dataset_path, train=False,
                                                      download=True, transform=transforms.ToTensor())
        elif dataset == 'imagenet':
            if self.args.use_admix == False:
                test_trans = torchvision.transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()]
                )
                self.data = torchvision.datasets.ImageNet(
                    root=args.dataset_path, split='val', transform=test_trans)

        else:
            raise ValueError('This dataset name does not exists')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_aa = self.data_aa[idx]
        if isinstance(self.data[idx], tuple):
            clear_sample = self.data[idx][0]
        else:
            clear_sample = self.data[idx]
        y = self.lables[idx]

        if self.transform and isinstance(sample_aa, torch.Tensor) == False:
            sample_aa = self.transform(sample_aa)

        if self.transform and isinstance(clear_sample, torch.Tensor) == False:
            clear_sample = self.transform(clear_sample)

        return sample_aa, y, clear_sample


def create_aa_dataset(model, args):
    """
    Generates adversarial examples for a specified model and saves them to a specified path.

    Parameters:
    - model: Model used for generating adversarial examples.
    - args: Contains batch size, attack parameters, and file paths for saving data.

    Notes:
    - Uses AutoAttack to generate adversarial examples.
    - Saves generated adversarial examples and labels to paths specified in args.
    """

    batch_size = args.batch_size
    args.batch_size = get_dataset_size(args)

    testloader = get_data_loader(args)
    args.batch_size = batch_size

    model.eval()
    x, y = next(iter(testloader))
    norm = args.attack_threat_model
    eps = args.attack_epsilon
    print('norm {},   epsilon {}'.format(norm, eps))

    # Initialize AutoAttack
    adversary = AutoAttack(model, norm=norm, eps=eps, version='standard', verbose=True)
    adversary.seed = args.seed
    x_adv = adversary.run_standard_evaluation(x, y, bs=args.batch_size)

    # Create directories and save adversarial examples and labels
    os.makedirs(os.path.dirname(args.aa_dataset_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.aa_labels_path), exist_ok=True)

    torch.save(x_adv, args.aa_dataset_path)
    torch.save(y, args.aa_labels_path)
