import math
import torch
from torch.optim.optimizer import Optimizer
import torch.nn as nn

class AdamDynamic(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, hypergrad_lr=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, hypergrad_lr=hypergrad_lr)
        super(AdamDynamic, self).__init__(params, defaults)

        # гиперпараметры как тензоры для обучения
        self.lr_hp    = nn.Parameter(torch.log(torch.tensor(1e-3)))
        self.beta1_hp = nn.Parameter(torch.tensor(9.99))
        self.beta2_hp = nn.Parameter(torch.tensor(9.99))
        self.opt_beta2 = True
        if self.opt_beta2:
            self.hyperparams = [self.lr_hp, self.beta1_hp, self.beta2_hp]
        else:
            self.hyperparams = [self.lr_hp, self.beta1_hp]

        # оптимизатор гиперпараметров
        self.opt_hparams = torch.optim.Adam(self.hyperparams, lr=hypergrad_lr)

    def step(self, closure=None):
        loss = None
        meta_loss = 0.0
        
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if not self.opt_beta2:
                    _, beta2 = group['betas']
                lr = torch.exp(self.lr_hp)
                beta1 = torch.sigmoid(self.beta1_hp) * 0.9
                if self.opt_beta2:
                    beta2 = torch.sigmoid(self.beta2_hp) * 0.999
                state["beta1_t"] = beta1.item()
                if self.opt_beta2:
                    state["beta2_t"] = beta2.item()
                else:
                    state["beta2_t"] = beta2
                group["lr"] = lr.item()

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * p

                # обычный Adam
                exp_avg = exp_avg * beta1 + grad * (1 - beta1)
                exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2)

                denom = torch.sqrt(exp_avg_sq) + group['eps']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                meta_loss += (step_size * exp_avg / denom).pow(2).mean()

                with torch.no_grad():
                    p.copy_(p - step_size * exp_avg / denom)

        self.opt_hparams.zero_grad()
        meta_loss.backward()
        self.opt_hparams.step()

        return loss
