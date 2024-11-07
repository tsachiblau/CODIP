import copy

import torch

from attack_utils import get_threat_model
from data_utils import get_data_loader


def eval_model(model, args):
    """
    Evaluates a model's performance on clean and adversarial examples for both a base classifier
    and the CODIP defense method.

    Parameters:
    - model: PyTorch model to be evaluated, which has an optional `use_defense` parameter to toggle CODIP.
    - args: Parsed command-line arguments containing model configuration, dataset, and defense settings.

    This function:
    - Loads the dataset using a data loader based on `args`.
    - Retrieves a threat model to generate adversarial examples using PGD attacks.
    - Initializes a dictionary `dict_eval` to track evaluation metrics, such as the number of correctly classified samples.
    - Iterates through the test data, evaluating the model's performance on:
        - Clean images without defense
        - Adversarially attacked images without defense
        - Clean images using the CODIP defense
        - Adversarially attacked images using the CODIP defense
    - Outputs cumulative results after each batch.

    Args:
        - testloader (DataLoader): A PyTorch DataLoader for the test dataset.
        - attack_pgd: A threat model used to generate adversarial examples.

    Metrics in `dict_eval`:
    - 'num_samples': Total number of samples evaluated.
    - 'clean_images_base_cls': Correct predictions on clean images without defense.
    - 'attacked_images_base_cls': Correct predictions on attacked images without defense.
    - 'clean_images_CODIP': Correct predictions on clean images with CODIP defense.
    - 'attcked_images_CODIP': Correct predictions on attacked images with CODIP defense.

    Prints:
    - Evaluation metrics after each batch, displaying counts for each metric.
    """

    # Initialize data loader and attack model
    testloader = get_data_loader(args)
    attack_pgd = get_threat_model(args)

    # Initialize evaluation metrics
    dict_eval = {}
    dict_eval['num_samples'] = 0
    dict_eval['clean_images_base_cls'] = 0
    dict_eval['attacked_images_base_cls'] = 0
    dict_eval['clean_images_CODIP'] = 0
    dict_eval['attcked_images_CODIP'] = 0

    # Iterate through test data batches
    for tuple_data in testloader:
        # Extract data tuple, handling different dataset structures
        if len(tuple_data) == 2:
            x, y = tuple_data
            x_clear = copy.copy(x)
        else:
            x, y, x_clear = tuple_data
        x, y, x_clear = x.cuda(), y.cuda(), x_clear.cuda()

        # Evaluate clean images using base classifier
        with torch.no_grad():
            bool_clean_images_base_cls = model(x_clear).argmax(dim=1).__eq__(y)

        # Generate adversarial examples and evaluate on base classifier
        x_attacked = attack_pgd.get_adv_x(model, x, y)
        with torch.no_grad():
            bool_attacked_images_base_cls = model(x_attacked).argmax(dim=1).__eq__(y)

        # Evaluate clean and attacked images using CODIP defense
        bool_clean_images_CODIP = model(x_clear, use_defense=True).argmax(dim=1).__eq__(y)
        bool_attcked_images_CODIP = model(x_attacked, use_defense=True).argmax(dim=1).__eq__(y)

        # Initialize evaluation metrics
        for i in range(len(y)):
            dict_eval['num_samples'] += 1
            dict_eval['clean_images_base_cls'] += bool_clean_images_base_cls[i].item()
            dict_eval['attacked_images_base_cls'] += bool_attacked_images_base_cls[i].item()
            dict_eval['clean_images_CODIP'] += bool_clean_images_CODIP[i].item()
            dict_eval['attcked_images_CODIP'] += bool_attcked_images_CODIP[i].item()

        str_disp = ''
        for key in dict_eval.keys():
            str_disp += key + ': ' + str(dict_eval[key]) + '\t'
        print(str_disp)
