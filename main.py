import argparse
import random

import numpy as np
import torch

from data_utils import create_aa_dataset
from defense_utils import add_defense
from eval_model import eval_model
from model_utils import load_model

parser = argparse.ArgumentParser(description='ReversePGD')

## general
parser.add_argument('--flow', type=str, default='eval',
                    help='Defines the flow of the program, e.g., evaluation or dataset creation.')
parser.add_argument('--seed', type=int, default=3407, metavar='S', help='Random seed for reproducibility.')
parser.add_argument('--num_workers', type=int, default=8, help='Number of worker threads for data loading')
parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='Batch size for processing data')
parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name, e.g., cifar10.')
parser.add_argument('--dataset_partition', type=str, default='test',
                    help='Dataset partition to use, e.g., test or train.')
parser.add_argument('--model_path', type=str, default='', help='Path to load the model from.')
parser.add_argument('--dataset_path', type=str, default='', help='Path to load the dataset from.')

## model
parser.add_argument('--net_name', default='at', help='Name of the network, e.g., adversarially trained (at).')
parser.add_argument('--net_arch', default='wrn', help='Network architecture, e.g., wide residual network (wrn).')
parser.add_argument('--net_depth', type=int, default=50, help='Depth of the network model.')
parser.add_argument('--net_width', type=int, default=0, help='Width factor of the network model.')

## attack
parser.add_argument('--attack_threat_model', default='Linf', help='Threat model for the attack, e.g., Linf or L2')
parser.add_argument('--attack_epsilon', type=float, default=8 / 255, help='Magnitude of perturbation for the attack.')
parser.add_argument('--attack_num_steps', default=20, help='Number of steps for the attack perturbation')
parser.add_argument('--attack_alpha', default=-1, help='Step size for the attack perturbation')
parser.add_argument('--aa_dataset_path', type=str, default=None, help='Path for AutoAttack dataset if applicable.')
parser.add_argument('--aa_labels_path', type=str, default=None, help='Path for AutoAttack labels if applicable.')

## defense
parser.add_argument('--defense_method', type=str, default='CODIP', help='Defense method to use, e.g., CODIP.')
parser.add_argument('--defense_num_steps', type=int, default=30, help='Number of steps for the defense perturbation.')
parser.add_argument('--defense_alpha', type=float, default=-1, help='Step size for the defense perturbation.')
parser.add_argument('--defense_gamma', type=float, default=1e-1, help='Regularization parameter for defense.')
parser.add_argument('--speed_up_ktop', type=int, default=-1,
                    help='Top-k speed-up factor for faster defense evaluation.')

args = parser.parse_args()

if __name__ == '__main__':

    if args.aa_dataset_path is not None:
        args.num_workers = 0

    args.attack_alpha = 2.5 * (args.attack_epsilon / args.attack_num_steps)

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load model based on given arguments
    model = load_model(args)

    if args.flow in ['eval']:
        model = add_defense(model, args)  # Apply the specified defense method
        eval_model(model, args)  # Evaluate the model on the dataset
    elif args.flow in ['create_aa']:
        create_aa_dataset(model, args)  # Generate a dataset for AutoAttack evaluation
