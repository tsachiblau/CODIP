import torch

from data_utils import get_num_of_classes


def add_defense(model, args):
    """
    Adds a defense mechanism to the model based on the specified defense method in args.

    Parameters:
    - model: PyTorch model to add the defense to.
    - args: Command-line arguments specifying the defense method.

    Returns:
    - An instance of the selected defense class (e.g., CODIP, Defense).

    Raises:
    - Exception: If an unrecognized defense method is provided.
    """
    if args.defense_method in ['CODIP']:
        return CODIP(model, args)
    elif args.defense_method in ['None']:
        return Defense(model, args)
    else:
        raise Exception('Unrecognized defense method')


def get_defense_threat_model(args):
    """
    Initializes and returns a defense threat model using targeted PGD with L2 distance.

    Parameters:
    - args: Command-line arguments specifying defense parameters.

    Returns:
    - Instance of `defense_targeted_pgd_l2`.
    """
    return defense_targeted_pgd_l2(args.defense_num_steps, args.defense_alpha, args.defense_gamma)


class Defense(torch.nn.Module):
    """
    Base class for a defense model.

    Attributes:
    - model: The PyTorch model to defend.
    - args: Command-line arguments for configuration.
    - N_classes: Number of classes in the dataset.

    Methods:
    - forward: Runs forward pass with or without defense.
    - get_defense_prediction_function: Returns model predictions for defended examples.
    """

    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.N_classes = get_num_of_classes(args)

    def forward(self, x, use_defense=False):
        if not use_defense:
            return self.model(x)
        else:
            return self.get_defense_prediction_function(x)

    def get_defense_prediction_function(self, x):
        return self.model(x)


class defense_targeted_pgd_linf():
    """
    Targeted PGD adversarial defense using Linf norm.

    Parameters:
    - epsilon: Maximum perturbation.
    - num_steps: Number of PGD steps.
    - alpha: Step size.
    - gamma: Regularization factor.
    - defense_distance: Distance metric.

    Methods:
    - get_adv_x: Generates adversarial examples with Linf norm constraints.
    """

    def __init__(self, epsilon, num_steps, alpha, gamma, defense_distance):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha
        self.gamma = gamma
        self.defense_distance = defense_distance

    def get_adv_x(self, model, x, y_target):
        delta = torch.zeros_like(x, requires_grad=True, device='cuda')

        for t in range(self.num_steps):
            loss1 = nn.CrossEntropyLoss()(model(x + delta), y_target)
            loss2 = self.gamma * nn.MSELoss()(delta, torch.zeros_like(delta))
            loss = loss1 + loss2
            loss.backward()
            delta.data = (delta.data - self.alpha * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
            delta.data = (x + delta.data).clamp(0, 1) - x
        return x + delta.detach()


class defense_targeted_pgd_l2():
    """
    Targeted PGD adversarial defense using L2 norm.

    Parameters:
    - num_steps: Number of PGD steps.
    - alpha: Step size.
    - gamma: Regularization factor.

    Methods:
    - norms: Computes the L2 norm across all but the first dimension.
    - get_adv_x: Generates adversarial examples with L2 norm constraints.
    """

    def __init__(self, num_steps, alpha, gamma):
        self.num_steps = num_steps
        self.alpha = alpha
        self.gamma = gamma
        self.distance_fn = torch.nn.MSELoss()

    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def get_adv_x(self, model, x, y_target):
        instance_dist = torch.zeros(len(x), ).cuda()
        delta = torch.zeros_like(x, requires_grad=True)
        for t in range(self.num_steps):
            loss1 = torch.nn.CrossEntropyLoss()(model(x + delta), y_target)
            loss2 = self.gamma * self.distance_fn(x, x + delta).mean()
            loss = loss1 + loss2
            loss.backward()
            norm = self.norms(delta.grad.detach())
            instance_dist += norm.view(len(x), )
            delta.data -= self.alpha * delta.grad.detach() / norm
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
            delta.grad.zero_()

        return x + delta.detach(), instance_dist


class CODIP(Defense):
    """
    CODIP defense class implementing conditional image transformation and distance-based prediction.

    Methods:
    - get_defense_prediction_function: Applies defense transformations and computes prediction scores based on MSE.
    """

    def __init__(self, model, args):
        super().__init__(model, args)
        self.defense_pgd = get_defense_threat_model(args)

    def get_defense_prediction_function(self, x):
        repeat_y = torch.Tensor([int(i) for i in range(self.N_classes)]).reshape((-1, 1)).repeat(
            (1, len(x))).flatten().long().cuda()
        repeat_x = x.repeat((self.N_classes, 1, 1, 1))
        ktop = self.args.speed_up_ktop

        tensor_prediction = torch.zeros((len(repeat_y),)).to(x.device)

        if self.args.speed_up_ktop > 0:
            with torch.no_grad():
                tensor_logits = self.model(x)

            tensor_argsort = tensor_logits.sort(dim=1, descending=True)[0]
            tendor_bool_calc = tensor_logits > (tensor_argsort[:, ktop].unsqueeze(1))
            tensor_bool_repreat = tendor_bool_calc.permute(1, 0).reshape(-1, )
        else:
            tensor_bool_repreat = torch.ones_like(repeat_y).bool()

        repeat_x, repeat_y = repeat_x[tensor_bool_repreat], repeat_y[tensor_bool_repreat]

        tensor_reverse_pgd, images_distance = self.defense_pgd.get_adv_x(self.model, repeat_x, repeat_y)

        tensor_prediction[tensor_bool_repreat] = 1 / (
                1e-10 + torch.nn.MSELoss(reduction='none')(tensor_reverse_pgd, repeat_x).sum(dim=(1, 2, 3)))
        MSE_pred = tensor_prediction.reshape(self.N_classes, -1).permute(1, 0).clone()

        return MSE_pred
