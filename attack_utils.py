import torch
from torch import nn


def get_threat_model(args):
    if args.aa_dataset_path is not None:
        attack = no_attack()
    else:
        if args.attack_threat_model == 'Linf':
            attack = pgd_linf(args.attack_epsilon, args.attack_num_steps, args.attack_alpha)
        elif args.attack_threat_model == 'L2':
            attack = pgd_l2(args.attack_epsilon, args.attack_num_steps, args.attack_alpha)
        else:
            raise ValueError('Unknown attack_threat_model value')

    return attack


class pgd_l2():
    def __init__(self, epsilon, num_steps, alpha):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha

    def norms(self, Z):
        """Compute norms over all but the first dimension"""
        return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]

    def get_adv_x(self, model, x, y):
        delta = torch.zeros_like(x, requires_grad=True, device='cuda')

        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(model(x + delta), y)
            loss.backward()
            delta.data += self.alpha * delta.grad.detach() / self.norms(delta.grad.detach())
            delta.data = torch.min(torch.max(delta.detach(), -x), 1 - x)  # clip X+delta to [0,1]
            delta.data *= self.epsilon / self.norms(delta.detach()).clamp(min=self.epsilon)
            delta.grad.zero_()
        return x + delta.detach()


class no_attack():
    def get_adv_x(self, model, x, y):
        return x


class pgd_linf():
    def __init__(self, epsilon, num_steps, alpha):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.alpha = alpha

    def get_adv_x(self, model, x, y):
        delta = torch.zeros_like(x, requires_grad=True, device='cuda')

        for t in range(self.num_steps):
            loss = nn.CrossEntropyLoss()(model(x + delta), y)
            loss.backward()
            delta.data = (delta.data + self.alpha * delta.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
            delta.data = (x + delta.data).clamp(0, 1) - x
        return x + delta.detach()
